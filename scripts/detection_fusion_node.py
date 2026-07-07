#!/usr/bin/env python3
"""
Fuses multi-camera YOLO 2D detections with lidar cluster objects to
produce semantically labeled 3D detections, then checks safety zones.

Design (v0.2):
- The lidar cluster node (obstacle_detection_node / _cuda) publishes
  TrackedObjectArray on /lidar/objects; this node no longer consumes the
  raw point cloud or visualization markers. Fusion runs once per lidar
  message (~10 Hz), bounding the CPU cost to O(objects x detections).
- Camera detections come from yolo_multi_cam on
  /yolo/<camera_name>/detections. Empty arrays matter: they refresh the
  per-camera cache so stale detections expire (detection_timeout guards
  against a dead camera node).
- Association is done in pixel space: lidar object centers are projected
  into each camera (honoring the camera's distortion model — the ArkCams
  are kannala_brandt fisheye) and matched against YOLO bboxes. A 3D
  nearest-neighbor pass on ground-projected camera detections catches
  what imperfect calibration misses.
- The camera's optical frame is taken from the Detection2DArray header
  (stamped by yolo_multi_cam from the image), so v1 bags and the live v2
  stack both work without frame configuration.

Safety-zone semantics are unchanged from v0.1 (including the current
x = lateral / y = forward deployment convention — do not "fix" this
without the planned HIL verification).
"""

import functools
import math
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import CameraInfo
from vision_msgs.msg import Detection2DArray

from obstacle_detection.msg import (
    SafetyAlert,
    SafetyAlertArray,
    TrackedObject,
    TrackedObjectArray,
    Zone,
    ZoneArray,
)

import tf2_ros

FISHEYE_MODELS = {"kannala_brandt", "equidistant", "fisheye"}


@dataclass
class CameraDetection:
    class_name: str
    confidence: float
    # bbox in pixels (center x/y, size x/y)
    cx: float
    cy: float
    w: float
    h: float


@dataclass
class CameraState:
    name: str
    frame_id: str | None = None          # from Detection2DArray header
    detections: list[CameraDetection] = field(default_factory=list)
    last_update: float = 0.0             # monotonic reception time
    # intrinsics
    K: np.ndarray | None = None
    D: np.ndarray | None = None
    dist_model: str = ""
    img_w: int = 0
    img_h: int = 0


class DetectionFusionNode(Node):
    def __init__(self):
        super().__init__("detection_fusion_node")

        self.declare_parameter("camera_names", ["zed_front"])
        self.declare_parameter("lidar_objects_topic", "/lidar/objects")
        self.declare_parameter("output_topic", "/safety_zone/detections")
        self.declare_parameter("alert_topic", "/safety_zone/alerts")
        self.declare_parameter("zone_config_topic", "/safety_zone/config")
        self.declare_parameter("target_frame", "base_link")

        self.declare_parameter("bbox_margin_pixels", 15)
        self.declare_parameter("max_association_distance", 2.0)
        self.declare_parameter("max_projection_range", 40.0)
        self.declare_parameter("detection_timeout", 0.7)
        self.declare_parameter("camera_only_merge_distance", 1.0)
        self.declare_parameter("enable_ground_projection_fallback", True)
        self.declare_parameter("ground_z", 0.0)

        # Zone convention note: x = lateral, y = forward (see module docstring).
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
        self._warned_frames: set[str] = set()

        qos_sensor = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=1)

        self.create_subscription(
            TrackedObjectArray, self.lidar_objects_topic, self._lidar_cb, 10
        )
        for cam in self._cameras.values():
            self.create_subscription(
                Detection2DArray,
                f"/yolo/{cam.name}/detections",
                functools.partial(self._yolo_cb, camera_name=cam.name),
                10,
            )
            for topic in self._camera_info_topics[cam.name]:
                self.create_subscription(
                    CameraInfo,
                    topic,
                    functools.partial(self._camera_info_cb, camera_name=cam.name),
                    qos_sensor,
                )
            self.get_logger().info(
                f"Camera '{cam.name}': /yolo/{cam.name}/detections, "
                f"info={self._camera_info_topics[cam.name]}"
            )

        self.det_pub = self.create_publisher(TrackedObjectArray, self.output_topic, 10)
        self.alert_pub = self.create_publisher(SafetyAlertArray, self.alert_topic, 10)

        zone_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            depth=1,
        )
        self.zone_pub = self.create_publisher(ZoneArray, self.zone_config_topic, zone_qos)
        self._publish_zone_config()

        self.get_logger().info(
            f"DetectionFusionNode started with {len(self._cameras)} camera(s)"
        )

    def _load_params(self):
        camera_names: list[str] = self.get_parameter("camera_names").value
        self.lidar_objects_topic = self.get_parameter("lidar_objects_topic").value
        self.output_topic = self.get_parameter("output_topic").value
        self.alert_topic = self.get_parameter("alert_topic").value
        self.zone_config_topic = self.get_parameter("zone_config_topic").value
        self.target_frame = self.get_parameter("target_frame").value

        self._cameras: dict[str, CameraState] = {}
        self._camera_info_topics: dict[str, list[str]] = {}
        for name in camera_names:
            prefix = f"cameras.{name}"
            self.declare_parameter(f"{prefix}.camera_info_topic", "")
            self.declare_parameter(f"{prefix}.camera_info_topic_v1", "")
            self.declare_parameter(f"{prefix}.camera_info_topic_v2", "")
            single = self.get_parameter(f"{prefix}.camera_info_topic").value
            v1 = self.get_parameter(f"{prefix}.camera_info_topic_v1").value
            v2 = self.get_parameter(f"{prefix}.camera_info_topic_v2").value
            topics = [single] if single else [t for t in dict.fromkeys([v2, v1]) if t]
            if not topics:
                self.get_logger().warn(f"Camera '{name}': no camera_info topic configured")
            self._cameras[name] = CameraState(name=name)
            self._camera_info_topics[name] = topics

        self.bbox_margin = self.get_parameter("bbox_margin_pixels").value
        self.max_assoc_dist = self.get_parameter("max_association_distance").value
        self.max_projection_range = self.get_parameter("max_projection_range").value
        self.detection_timeout = self.get_parameter("detection_timeout").value
        self.camera_only_merge_distance = self.get_parameter(
            "camera_only_merge_distance").value
        self.enable_ground_projection_fallback = self.get_parameter(
            "enable_ground_projection_fallback").value
        self.ground_z = self.get_parameter("ground_z").value

        self.zones = {
            "red": {
                "x_min": self.get_parameter("red_zone_x_min").value,
                "x_max": self.get_parameter("red_zone_x_max").value,
                "y_min": self.get_parameter("red_zone_y_min").value,
                "y_max": self.get_parameter("red_zone_y_max").value,
            },
            "yellow": {
                "x_min": self.get_parameter("yellow_zone_x_min").value,
                "x_max": self.get_parameter("yellow_zone_x_max").value,
                "y_min": self.get_parameter("yellow_zone_y_min").value,
                "y_max": self.get_parameter("yellow_zone_y_max").value,
            },
        }

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _camera_info_cb(self, msg: CameraInfo, *, camera_name: str):
        cam = self._cameras[camera_name]
        cam.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        cam.D = np.array(msg.d, dtype=np.float64)
        cam.dist_model = msg.distortion_model.lower()
        cam.img_w = msg.width
        cam.img_h = msg.height

    def _yolo_cb(self, msg: Detection2DArray, *, camera_name: str):
        cam = self._cameras[camera_name]
        if msg.header.frame_id:
            cam.frame_id = msg.header.frame_id
        cam.last_update = time.monotonic()
        cam.detections = [
            CameraDetection(
                class_name=det.results[0].hypothesis.class_id,
                confidence=det.results[0].hypothesis.score,
                cx=det.bbox.center.position.x,
                cy=det.bbox.center.position.y,
                w=det.bbox.size_x,
                h=det.bbox.size_y,
            )
            for det in msg.detections
            if det.results
        ]

    def _lidar_cb(self, msg: TrackedObjectArray):
        if msg.header.frame_id != self.target_frame and \
                msg.header.frame_id not in self._warned_frames:
            self._warned_frames.add(msg.header.frame_id)
            self.get_logger().warn(
                f"Lidar objects arrive in frame '{msg.header.frame_id}' but "
                f"target_frame is '{self.target_frame}' — using the lidar frame. "
                "Safety zones are checked in the lidar objects' frame."
            )

        fused = self._fuse(list(msg.objects), msg.header.frame_id)

        out = TrackedObjectArray()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = msg.header.frame_id
        out.objects = fused
        self.det_pub.publish(out)

        alerts = self._check_safety_zones(fused, out.header)
        self.alert_pub.publish(alerts)

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    def _fuse(self, lidar_objs: list[TrackedObject], lidar_frame: str
              ) -> list[TrackedObject]:
        now = time.monotonic()
        centers = np.array(
            [[o.position.x, o.position.y, o.position.z] for o in lidar_objs],
            dtype=np.float64,
        ).reshape(-1, 3)

        # class label per lidar object index: (class_name, confidence)
        labels: dict[int, tuple[str, float]] = {}
        # camera detections that matched nothing in pixel space:
        # (CameraDetection, ground position or None)
        leftovers: list[tuple[CameraDetection, np.ndarray | None]] = []

        for cam in self._cameras.values():
            if now - cam.last_update > self.detection_timeout:
                continue  # stale or never seen — contributes nothing
            if not cam.detections:
                continue
            if cam.K is None or cam.frame_id is None:
                self.get_logger().warn(
                    f"Camera '{cam.name}': detections but no CameraInfo/frame yet",
                    throttle_duration_sec=5.0,
                )
                continue

            # Maps lidar-frame points into the camera optical frame.
            transform = self._lookup_transform(cam.frame_id, lidar_frame)
            if transform is None:
                continue
            rot, trans = self._tf_to_rot_trans(transform)
            cam_pts = (rot @ centers.T).T + trans

            uv, valid = self._project(cam, cam_pts)

            for det in cam.detections:
                margin = self.bbox_margin
                u_min = det.cx - det.w / 2.0 - margin
                u_max = det.cx + det.w / 2.0 + margin
                v_min = det.cy - det.h / 2.0 - margin
                v_max = det.cy + det.h / 2.0 + margin

                best_idx = -1
                best_range = float("inf")
                for i in range(len(lidar_objs)):
                    if not valid[i]:
                        continue
                    if not (u_min <= uv[i, 0] <= u_max and v_min <= uv[i, 1] <= v_max):
                        continue
                    rng = cam_pts[i, 2]  # closest object wins (occlusion)
                    if rng < best_range:
                        best_range = rng
                        best_idx = i

                if best_idx >= 0:
                    prev = labels.get(best_idx)
                    if prev is None or det.confidence > prev[1]:
                        labels[best_idx] = (det.class_name, det.confidence)
                else:
                    ground = None
                    if self.enable_ground_projection_fallback:
                        # Ray needs camera->lidar: invert the transform.
                        ground = self._ground_position(
                            det, cam, rot.T, -rot.T @ trans)
                    leftovers.append((det, ground))

        # Second chance: 3D nearest neighbor between ground-projected
        # camera detections and still-unlabeled lidar objects.
        camera_only: list[tuple[CameraDetection, np.ndarray]] = []
        for det, ground in leftovers:
            if ground is None:
                continue
            best_idx = -1
            best_dist = self.max_assoc_dist
            for i, obj in enumerate(lidar_objs):
                if i in labels:
                    continue
                dist = math.hypot(ground[0] - obj.position.x,
                                  ground[1] - obj.position.y)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i
            if best_idx >= 0:
                labels[best_idx] = (det.class_name, det.confidence)
            else:
                camera_only.append((det, ground))

        fused: list[TrackedObject] = []
        for i, obj in enumerate(lidar_objs):
            out = TrackedObject()
            out.track_id = obj.track_id
            out.position = obj.position
            out.size = obj.size
            out.velocity = obj.velocity
            if i in labels:
                out.class_name, out.confidence = labels[i]
                out.source = "fused"
            else:
                out.class_name = obj.class_name or "unknown"
                out.confidence = obj.confidence
                out.source = obj.source or "lidar"
            fused.append(out)

        # Camera-only detections: dedupe across overlapping cameras and
        # against already-output objects.
        accepted: list[np.ndarray] = [
            np.array([o.position.x, o.position.y]) for o in fused
        ]
        for det, ground in sorted(camera_only, key=lambda t: -t[0].confidence):
            pos2 = np.array([ground[0], ground[1]])
            if any(np.hypot(pos2[0] - a[0], pos2[1] - a[1])
                   < self.camera_only_merge_distance for a in accepted):
                continue
            accepted.append(pos2)
            out = TrackedObject()
            out.track_id = -1
            out.class_name = det.class_name
            out.confidence = det.confidence
            out.position = Point(x=float(ground[0]), y=float(ground[1]),
                                 z=float(ground[2]))
            out.size = Vector3(x=0.5, y=0.5, z=1.7)
            out.velocity = Vector3()
            out.source = "camera"
            fused.append(out)

        return fused

    # ------------------------------------------------------------------
    # Projection helpers
    # ------------------------------------------------------------------

    def _project(self, cam: CameraState, cam_pts: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray]:
        """Project camera-frame points to (distorted) pixels.

        Returns (uv (N,2), valid (N,) bool). Points behind the camera or
        beyond max_projection_range are invalid.
        """
        n = cam_pts.shape[0]
        uv = np.zeros((n, 2), dtype=np.float64)
        valid = (cam_pts[:, 2] > 0.1) & \
                (np.linalg.norm(cam_pts, axis=1) <= self.max_projection_range)
        if not valid.any():
            return uv, valid

        pts = cam_pts[valid].reshape(-1, 1, 3)
        if cam.dist_model in FISHEYE_MODELS:
            d = cam.D[:4].reshape(4, 1) if cam.D.size >= 4 else np.zeros((4, 1))
            proj, _ = cv2.fisheye.projectPoints(
                pts, np.zeros(3), np.zeros(3), cam.K, d)
        else:
            d = cam.D if cam.D.size else np.zeros(5)
            proj, _ = cv2.projectPoints(
                pts, np.zeros(3), np.zeros(3), cam.K, d)
        uv[valid] = proj.reshape(-1, 2)

        # A projection far outside the image cannot match any bbox; also
        # guards against distortion extrapolation artifacts far off-axis.
        if cam.img_w and cam.img_h:
            in_img = (uv[:, 0] > -cam.img_w) & (uv[:, 0] < 2 * cam.img_w) & \
                     (uv[:, 1] > -cam.img_h) & (uv[:, 1] < 2 * cam.img_h)
            valid &= in_img
        return uv, valid

    def _ground_position(self, det: CameraDetection, cam: CameraState,
                         rot_cam_to_lidar: np.ndarray,
                         cam_origin_lidar: np.ndarray) -> np.ndarray | None:
        """Intersect the bbox bottom-center ray with z = ground_z (lidar frame)."""
        px = np.array([[[det.cx, det.cy + det.h / 2.0]]], dtype=np.float64)
        try:
            if cam.dist_model in FISHEYE_MODELS:
                d = cam.D[:4].reshape(4, 1) if cam.D.size >= 4 else np.zeros((4, 1))
                norm = cv2.fisheye.undistortPoints(px, cam.K, d)
            else:
                d = cam.D if cam.D.size else np.zeros(5)
                norm = cv2.undistortPoints(px, cam.K, d)
        except cv2.error:
            return None

        ray_cam = np.array([norm[0, 0, 0], norm[0, 0, 1], 1.0])
        ray_cam /= np.linalg.norm(ray_cam)
        ray = rot_cam_to_lidar @ ray_cam
        origin = cam_origin_lidar

        if abs(ray[2]) < 1e-6:
            return None
        distance = (self.ground_z - origin[2]) / ray[2]
        if distance <= 0.0 or distance > self.max_projection_range:
            return None
        return origin + distance * ray

    # ------------------------------------------------------------------
    # TF helpers
    # ------------------------------------------------------------------

    def _lookup_transform(self, target_frame: str, source_frame: str):
        """Transform mapping source-frame points into target frame."""
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

    def _tf_to_rot_trans(self, tf_stamped) -> tuple[np.ndarray, np.ndarray]:
        t = tf_stamped.transform.translation
        q = tf_stamped.transform.rotation
        trans = np.array([t.x, t.y, t.z])
        rot = self._quat_to_matrix(q.x, q.y, q.z, q.w)
        return rot, trans

    @staticmethod
    def _quat_to_matrix(qx, qy, qz, qw) -> np.ndarray:
        return np.array([
            [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ])

    # ------------------------------------------------------------------
    # Safety zones
    # ------------------------------------------------------------------

    def _publish_zone_config(self):
        cfg = ZoneArray()
        cfg.header.stamp = self.get_clock().now().to_msg()
        cfg.header.frame_id = self.target_frame
        for name, z in self.zones.items():
            zone = Zone()
            zone.name = name
            zone.x_min = float(z["x_min"])
            zone.x_max = float(z["x_max"])
            zone.y_min = float(z["y_min"])
            zone.y_max = float(z["y_max"])
            cfg.zones.append(zone)
        self.zone_pub.publish(cfg)

    def _check_safety_zones(self, objects, header) -> SafetyAlertArray:
        alert_array = SafetyAlertArray()
        alert_array.header = header

        for obj in objects:
            zone = self._point_in_zone(obj.position.x, obj.position.y)
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
        for name in ("red", "yellow"):
            z = self.zones[name]
            if z["x_min"] <= x <= z["x_max"] and z["y_min"] <= y <= z["y_max"]:
                return name
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
