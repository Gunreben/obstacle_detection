#!/usr/bin/env python3
"""
Fuses front-facing YOLO 2D detections with Ouster LiDAR depth to produce
semantically labeled 3D detections in base_link, then checks safety zones.
"""

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


class DetectionFusionNode(Node):
    def __init__(self):
        super().__init__("detection_fusion_node")

        self.declare_parameter("yolo_topic", "/zed/zed_node/left_raw/image_raw_color/compressed_bboxes")
        self.declare_parameter("marker_topic", "/obstacle_markers")
        self.declare_parameter("pointcloud_topic", "/ouster/points")
        self.declare_parameter("output_topic", "/safety_zone/detections")
        self.declare_parameter("alert_topic", "/safety_zone/alerts")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("camera_frame", "zed_left_camera_optical_frame")
        self.declare_parameter("camera_info_topic", "/zed/left/camera_info")

        self.declare_parameter("bbox_margin_pixels", 15)
        self.declare_parameter("max_association_distance", 2.0)
        self.declare_parameter("min_lidar_points_in_bbox", 3)

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
        self.create_subscription(
            Detection2DArray, self.yolo_topic, self._yolo_cb, 10
        )
        self.create_subscription(
            CameraInfo, self.camera_info_topic, self._camera_info_cb, qos_sensor
        )

        self.det_pub = self.create_publisher(
            TrackedObjectArray, self.output_topic, 10
        )
        self.alert_pub = self.create_publisher(
            SafetyAlertArray, self.alert_topic, 10
        )

        self.get_logger().info("DetectionFusionNode started")

    def _load_params(self):
        self.yolo_topic = self.get_parameter("yolo_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.pointcloud_topic = self.get_parameter("pointcloud_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.alert_topic = self.get_parameter("alert_topic").value
        self.target_frame = self.get_parameter("target_frame").value
        self.camera_frame = self.get_parameter("camera_frame").value
        self.camera_info_topic = self.get_parameter("camera_info_topic").value

        self.fx: float | None = None
        self.fy: float | None = None
        self.cx: float | None = None
        self.cy: float | None = None
        self.img_w: int | None = None
        self.img_h: int | None = None

        self.bbox_margin = self.get_parameter("bbox_margin_pixels").value
        self.max_assoc_dist = self.get_parameter("max_association_distance").value
        self.min_pts = self.get_parameter("min_lidar_points_in_bbox").value

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

    def _camera_info_cb(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.img_w = msg.width
        self.img_h = msg.height

    def _cloud_cb(self, msg: PointCloud2):
        self._latest_cloud = msg

    def _marker_cb(self, msg: MarkerArray):
        self._latest_markers = msg

    def _yolo_cb(self, msg: Detection2DArray):
        if self.fx is None:
            self.get_logger().warn(
                f"No CameraInfo received on {self.camera_info_topic} yet, skipping",
                throttle_duration_sec=5.0,
            )
            return

        cloud = self._latest_cloud
        if cloud is None:
            return

        cam_tf = self._lookup_transform(cloud.header.frame_id, self.camera_frame)
        base_tf = self._lookup_transform(cloud.header.frame_id, self.target_frame)
        if cam_tf is None or base_tf is None:
            return

        cloud_xyz = self._cloud_to_numpy(cloud)
        if cloud_xyz is None or cloud_xyz.shape[0] == 0:
            return

        cam_points = self._transform_points(cloud_xyz, cam_tf)
        base_points = self._transform_points(cloud_xyz, base_tf)

        camera_detections = self._project_yolo_to_3d(
            msg.detections, cam_points, base_points
        )

        lidar_objects = self._markers_to_objects()

        fused = self._fuse_detections(camera_detections, lidar_objects)

        out = TrackedObjectArray()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.target_frame
        out.objects = fused

        self.det_pub.publish(out)

        alerts = self._check_safety_zones(fused, out.header)
        if alerts.alerts:
            self.alert_pub.publish(alerts)

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

    def _project_yolo_to_3d(self, detections, cam_points, base_points):
        results = []

        in_front = cam_points[:, 2] > 0.1
        cam_front = cam_points[in_front]
        base_front = base_points[in_front]

        if cam_front.shape[0] == 0:
            return results

        u = (self.fx * cam_front[:, 0] / cam_front[:, 2]) + self.cx
        v = (self.fy * cam_front[:, 1] / cam_front[:, 2]) + self.cy

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
                continue

            median_pos = np.median(pts_in_box, axis=0)

            obj = TrackedObject()
            obj.track_id = -1
            obj.class_name = cls_name
            obj.confidence = conf
            obj.position = Point(x=median_pos[0], y=median_pos[1], z=median_pos[2])
            obj.size = Vector3(x=0.5, y=0.5, z=1.7)
            obj.velocity = Vector3(x=0.0, y=0.0, z=0.0)
            obj.source = "camera"
            results.append(obj)

        return results

    def _markers_to_objects(self) -> list[TrackedObject]:
        objects = []
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

    def _fuse_detections(
        self,
        camera_dets: list[TrackedObject],
        lidar_objs: list[TrackedObject],
    ) -> list[TrackedObject]:
        lidar_matched = [False] * len(lidar_objs)
        fused = []

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
