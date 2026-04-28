#!/usr/bin/env python3
"""
Fuses multi-camera YOLO 2D detections with Ouster LiDAR depth to produce
semantically labeled 3D detections in base_link, then checks safety zones.

Supports an arbitrary number of cameras configured via the ``camera_names``
and ``cameras.<name>.*`` parameters.
"""

import functools
from dataclasses import dataclass, field

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import CameraInfo, PointCloud2
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import MarkerArray

from obstacle_detection.msg import (
    SafetyAlert,
    SafetyAlertArray,
    TrackedObject,
    TrackedObjectArray,
)

import tf2_ros
import sensor_msgs_py.point_cloud2 as pc2


@dataclass
class CameraState:
    name: str
    yolo_topic: str
    camera_frame: str
    camera_info_topic: str
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    img_w: int | None = None
    img_h: int | None = None
    latest_detections: list[TrackedObject] = field(default_factory=list)


class DetectionFusionNode(Node):
    def __init__(self):
        super().__init__("detection_fusion_node")

        self.declare_parameter("camera_names", ["zed_front"])
        self.declare_parameter("marker_topic", "/obstacle_markers")
        self.declare_parameter("pointcloud_topic", "/ouster/points")
        self.declare_parameter("output_topic", "/safety_zone/detections")
        self.declare_parameter("alert_topic", "/safety_zone/alerts")
        self.declare_parameter("target_frame", "base_link")

        self.declare_parameter("bbox_margin_pixels", 15)
        self.declare_parameter("max_association_distance", 2.0)
        self.declare_parameter("min_lidar_points_in_bbox", 3)
        self.declare_parameter("enable_ground_projection_fallback", True)
        self.declare_parameter("ground_z", 0.0)

        self.declare_parameter("red_zone_x_min", -2.0)
        self.declare_parameter("red_zone_x_max", 2.0)
        self.declare_parameter("red_zone_y_min", -4.0)
        self.declare_parameter("red_zone_y_max", 6.0)
        self.declare_parameter("yellow_zone_x_min", -4.0)
        self.declare_parameter("yellow_zone_x_max", 4.0)
        self.declare_parameter("yellow_zone_y_min", -6.0)
        self.declare_parameter("yellow_zone_y_max", 10.0)

        self._load_params()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._latest_cloud: PointCloud2 | None = None
        self._latest_markers: MarkerArray | None = None

        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)

        self.create_subscription(
            PointCloud2, self.pointcloud_topic, self._cloud_cb, qos_sensor
        )
        self.create_subscription(
            MarkerArray, self.marker_topic, self._marker_cb, 10
        )

        for cam in self._cameras.values():
            self.create_subscription(
                Detection2DArray,
                cam.yolo_topic,
                functools.partial(self._yolo_cb, camera_name=cam.name),
                10,
            )
            self.create_subscription(
                CameraInfo,
                cam.camera_info_topic,
                functools.partial(self._camera_info_cb, camera_name=cam.name),
                qos_sensor,
            )
            self.get_logger().info(
                f"Camera '{cam.name}': yolo={cam.yolo_topic}  "
                f"frame={cam.camera_frame}  info={cam.camera_info_topic}"
            )

        self.det_pub = self.create_publisher(
            TrackedObjectArray, self.output_topic, 10
        )
        self.alert_pub = self.create_publisher(
            SafetyAlertArray, self.alert_topic, 10
        )

        self.get_logger().info(
            f"DetectionFusionNode started with {len(self._cameras)} camera(s)"
        )

    def _load_params(self):
        camera_names: list[str] = self.get_parameter("camera_names").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.alert_topic = self.get_parameter("alert_topic").value
        self.target_frame = self.get_parameter("target_frame").value

        self._cameras: dict[str, CameraState] = {}
        for name in camera_names:
            prefix = f"cameras.{name}"
            self.declare_parameter(f"{prefix}.yolo_topic", "")
            self.declare_parameter(f"{prefix}.camera_frame", "")
            self.declare_parameter(f"{prefix}.camera_info_topic", "")

            self._cameras[name] = CameraState(
                name=name,
                yolo_topic=self.get_parameter(f"{prefix}.yolo_topic").value,
                camera_frame=self.get_parameter(f"{prefix}.camera_frame").value,
                camera_info_topic=self.get_parameter(f"{prefix}.camera_info_topic").value,
            )

        self.bbox_margin = self.get_parameter("bbox_margin_pixels").value
        self.max_assoc_dist = self.get_parameter("max_association_distance").value
        self.min_pts = self.get_parameter("min_lidar_points_in_bbox").value
        self.enable_ground_projection_fallback = self.get_parameter(
            "enable_ground_projection_fallback"
        ).value
        self.ground_z = self.get_parameter("ground_z").value

        self.red_zone = {
            "x_min": self.get_parameter("red_zone_x_min").value,
            "x_max": self.get_parameter("red_zone_x_max").value,
            "y_min": self.get_parameter("red_zone_y_min").value,
            "y_max": self.get_parameter("red_zone_y_max").value,
        }
        self.yellow_zone = {
            "x_min": self.get_parameter("yellow_zone_x_min").value,
            "x_max": self.get_parameter("yellow_zone_x_max").value,
            "y_min": self.get_parameter("yellow_zone_y_min").value,
            "y_max": self.get_parameter("yellow_zone_y_max").value,
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo, *, camera_name: str):
        cam = self._cameras[camera_name]
        cam.fx = msg.k[0]
        cam.fy = msg.k[4]
        cam.cx = msg.k[2]
        cam.cy = msg.k[5]
        cam.img_w = msg.width
        cam.img_h = msg.height

    def _cloud_cb(self, msg: PointCloud2):
        self._latest_cloud = msg

    def _marker_cb(self, msg: MarkerArray):
        self._latest_markers = msg

    def _yolo_cb(self, msg: Detection2DArray, *, camera_name: str):
        cam = self._cameras[camera_name]
        if cam.fx is None:
            self.get_logger().warn(
                f"No CameraInfo for '{camera_name}' on "
                f"{cam.camera_info_topic} yet, skipping",
                throttle_duration_sec=5.0,
            )
            return

        cloud = self._latest_cloud
        if cloud is None:
            return

        cam_tf = self._lookup_transform(cloud.header.frame_id, cam.camera_frame)
        base_tf = self._lookup_transform(cloud.header.frame_id, self.target_frame)
        cam_to_base_tf = self._lookup_transform(cam.camera_frame, self.target_frame)
        if cam_tf is None or base_tf is None or cam_to_base_tf is None:
            return

        cloud_xyz = self._cloud_to_numpy(cloud)
        if cloud_xyz is None or cloud_xyz.shape[0] == 0:
            return

        cam_points = self._transform_points(cloud_xyz, cam_tf)
        base_points = self._transform_points(cloud_xyz, base_tf)

        cam.latest_detections = self._project_yolo_to_3d(
            msg.detections, cam_points, base_points, cam, cam_to_base_tf
        )

        all_camera_dets: list[TrackedObject] = []
        for c in self._cameras.values():
            all_camera_dets.extend(c.latest_detections)

        lidar_objects = self._markers_to_objects()
        fused = self._fuse_detections(all_camera_dets, lidar_objects)

        out = TrackedObjectArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.target_frame
        out.objects = fused

        self.det_pub.publish(out)

        alerts = self._check_safety_zones(fused, out.header)
        if alerts.alerts:
            self.alert_pub.publish(alerts)

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------

    def _lookup_transform(self, source_frame: str, target_frame: str):
        try:
            return self.tf_buffer.lookup_transform(
                target_frame, source_frame, rclpy.time.Time()
            )
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ) as e:
            self.get_logger().warn(
                f"TF lookup {source_frame} -> {target_frame}: {e}",
                throttle_duration_sec=2.0,
            )
            return None

    def _cloud_to_numpy(self, cloud: PointCloud2) -> np.ndarray | None:
        try:
            structured = pc2.read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)
            arr = np.array(structured)
            if arr.ndim == 0 or arr.shape[0] == 0:
                return None
            if arr.dtype.names is not None:
                xyz = np.column_stack([arr[f].astype(np.float64) for f in ("x", "y", "z")])
            else:
                xyz = arr.astype(np.float64)
            return xyz
        except Exception as e:
            self.get_logger().warn(f"Failed to parse cloud: {e}", throttle_duration_sec=2.0)
            return None

    def _transform_points(self, points: np.ndarray, tf_stamped) -> np.ndarray:
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        trans = np.array([t.x, t.y, t.z])
        rot = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        return (rot @ points.T).T + trans

    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw) -> np.ndarray:
        r = np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])
        return r

    # ------------------------------------------------------------------
    # 3D projection
    # ------------------------------------------------------------------

    def _project_yolo_to_3d(
        self,
        detections,
        cam_points: np.ndarray,
        base_points: np.ndarray,
        cam: CameraState,
        cam_to_base_tf,
    ) -> list[TrackedObject]:
        results: list[TrackedObject] = []

        in_front = cam_points[:, 2] > 0.1
        cam_front = cam_points[in_front]
        base_front = base_points[in_front]

        if cam_front.shape[0] == 0:
            return results

        u = (cam.fx * cam_front[:, 0] / cam_front[:, 2]) + cam.cx
        v = (cam.fy * cam_front[:, 1] / cam_front[:, 2]) + cam.cy

        for det in detections:
            if not det.results:
                continue

            cls_name = det.results[0].hypothesis.class_id
            conf = det.results[0].hypothesis.score

            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            w = det.bbox.size_x
            h = det.bbox.size_y

            margin = self.bbox_margin
            u_min = cx - w / 2.0 - margin
            u_max = cx + w / 2.0 + margin
            v_min = cy - h / 2.0 - margin
            v_max = cy + h / 2.0 + margin

            mask = (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max)
            pts_in_box = base_front[mask]

            if pts_in_box.shape[0] < self.min_pts:
                if not self.enable_ground_projection_fallback:
                    continue
                fallback_pos = self._project_detection_to_ground(det, cam, cam_to_base_tf)
                if fallback_pos is None:
                    continue
                pos = fallback_pos
            else:
                pos = np.median(pts_in_box, axis=0)

            obj = TrackedObject()
            obj.track_id = -1
            obj.class_name = cls_name
            obj.confidence = conf
            obj.position = Point(x=pos[0], y=pos[1], z=pos[2])
            obj.size = Vector3(x=0.5, y=0.5, z=1.7)
            obj.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            obj.source = "camera"
            results.append(obj)

        return results

    def _project_detection_to_ground(self, det, cam: CameraState, cam_to_base_tf):
        """Project bbox bottom-center through an optical frame onto z=ground_z."""
        u = det.bbox.center.position.x
        v = det.bbox.center.position.y + det.bbox.size_y / 2.0

        ray_cam = np.array([(u - cam.cx) / cam.fx, (v - cam.cy) / cam.fy, 1.0])
        ray_cam = ray_cam / np.linalg.norm(ray_cam)

        t = cam_to_base_tf.transform.translation
        q = cam_to_base_tf.transform.rotation
        origin_base = np.array([t.x, t.y, t.z])
        ray_base = self._quat_to_matrix(q.x, q.y, q.z, q.w) @ ray_cam

        if abs(ray_base[2]) < 1e-6:
            return None

        distance = (self.ground_z - origin_base[2]) / ray_base[2]
        if distance <= 0.0:
            return None

        return origin_base + distance * ray_base

    # ------------------------------------------------------------------
    # LiDAR markers -> TrackedObjects
    # ------------------------------------------------------------------

    def _markers_to_objects(self) -> list[TrackedObject]:
        objects: list[TrackedObject] = []
        if self._latest_markers is None:
            return objects

        for marker in self._latest_markers.markers:
            if marker.action != 0:
                continue

            obj = TrackedObject()
            obj.track_id = marker.id
            obj.class_name = "unknown"
            obj.confidence = 0.0
            obj.position = Point(
                x=marker.pose.position.x,
                y=marker.pose.position.y,
                z=marker.pose.position.z,
            )
            obj.size = Vector3(
                x=marker.scale.x, y=marker.scale.y, z=marker.scale.z
            )
            obj.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            obj.source = "lidar"
            objects.append(obj)

        return objects

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse_detections(
        self,
        camera_dets: list[TrackedObject],
        lidar_objs: list[TrackedObject],
    ) -> list[TrackedObject]:
        lidar_matched = [False] * len(lidar_objs)
        fused: list[TrackedObject] = []

        for cam_det in camera_dets:
            best_idx = -1
            best_dist = self.max_assoc_dist

            for i, lidar_obj in enumerate(lidar_objs):
                if lidar_matched[i]:
                    continue
                dist = np.sqrt(
                    (cam_det.position.x - lidar_obj.position.x) ** 2
                    + (cam_det.position.y - lidar_obj.position.y) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx >= 0:
                lidar_matched[best_idx] = True
                merged = TrackedObject()
                merged.track_id = lidar_objs[best_idx].track_id
                merged.class_name = cam_det.class_name
                merged.confidence = cam_det.confidence
                merged.position = lidar_objs[best_idx].position
                merged.size = lidar_objs[best_idx].size
                merged.velocity = lidar_objs[best_idx].velocity
                merged.source = "fused"
                fused.append(merged)
            else:
                fused.append(cam_det)

        for i, lidar_obj in enumerate(lidar_objs):
            if not lidar_matched[i]:
                fused.append(lidar_obj)

        return fused

    # ------------------------------------------------------------------
    # Safety zones
    # ------------------------------------------------------------------

    def _check_safety_zones(self, objects, header) -> SafetyAlertArray:
        alert_array = SafetyAlertArray()
        alert_array.header = header

        for obj in objects:
            x = obj.position.x
            y = obj.position.y

            zone = self._point_in_zone(x, y)
            if zone is not None:
                alert = SafetyAlert()
                alert.header = header
                alert.zone_name = zone
                alert.track_id = obj.track_id
                alert.class_name = obj.class_name
                alert.position = obj.position
                alert_array.alerts.append(alert)

        return alert_array

    def _point_in_zone(self, x: float, y: float) -> str | None:
        rz = self.red_zone
        if rz["x_min"] <= x <= rz["x_max"] and rz["y_min"] <= y <= rz["y_max"]:
            return "red"
        yz = self.yellow_zone
        if yz["x_min"] <= x <= yz["x_max"] and yz["y_min"] <= y <= yz["y_max"]:
            return "yellow"
        return None


def main(args=None):
    rclpy.init(args=args)
    node = DetectionFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
