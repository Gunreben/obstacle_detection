#!/usr/bin/env python3
"""Safety zones + tracked objects as a Foxglove 3D scene (foxglove_msgs/SceneUpdate).

Renders what detection_fusion_node publishes into the Foxglove 3D panel,
on top of whatever else is in the scene (e.g. the semantic BEV ground grid):

- zones (/safety_zone/config): translucent ground rectangles with outline
  and STOP / WARNING labels; a zone lights up while an object is inside.
- objects (/safety_zone/detections): boxes with edges + billboard label
  "<class> #<track> <dist> m" (no label for far unknown clusters). Colour by class, red / amber when inside the
  red / yellow zone; camera-only objects (no lidar support) are drawn
  fainter and labelled "(cam)".
- optional status text above the vehicle while the red zone is occupied.

Zone membership is recomputed here with the same rule as the fusion node
(red first, then yellow) from the latched zone config, so colours always
match the published zones.

Output: /safety_zone/scene (one SceneUpdate per detections message; the
object entity has a short lifetime so it disappears when the pipeline stops).
"""

import math

import rclpy
from builtin_interfaces.msg import Duration
from foxglove_msgs.msg import (
    Color,
    CubePrimitive,
    LinePrimitive,
    SceneEntity,
    SceneUpdate,
    TextPrimitive,
    TriangleListPrimitive,
)
from geometry_msgs.msg import Point, Pose, Vector3
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from obstacle_detection.msg import TrackedObjectArray, ZoneArray

ZONE_RGB = {"red": (1.0, 0.15, 0.15), "yellow": (1.0, 0.78, 0.0)}
ZONE_LABEL = {"red": "STOP", "yellow": "WARNING"}
CLASS_RGB = {
    "person": (0.0, 1.0, 0.53),
    "unknown": (0.55, 0.65, 0.8),
}
DEFAULT_RGB = (0.3, 0.75, 1.0)


def _color(rgb, a):
    return Color(r=float(rgb[0]), g=float(rgb[1]), b=float(rgb[2]), a=float(a))


def _pose(x=0.0, y=0.0, z=0.0):
    p = Pose()
    p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
    p.orientation.w = 1.0
    return p


def _box_edges(cx, cy, cz, sx, sy, sz):
    """12 box edges as LINE_LIST points."""
    xs, ys, zs = (cx - sx / 2, cx + sx / 2), (cy - sy / 2, cy + sy / 2), (cz - sz / 2, cz + sz / 2)
    c = [Point(x=x, y=y, z=z) for z in zs for y in ys for x in xs]  # index = 4z + 2y + x
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    return [p for e in edges for p in (c[e[0]], c[e[1]])]


class SafetyZoneSceneNode(Node):
    def __init__(self):
        super().__init__("safety_zone_scene_node")
        self.declare_parameter("detections_topic", "/safety_zone/detections")
        self.declare_parameter("zone_config_topic", "/safety_zone/config")
        self.declare_parameter("scene_topic", "/safety_zone/scene")
        # Height of the drawn ground (match the BEV grid; slightly above it
        # so the zones are not z-fighting with the grid).
        self.declare_parameter("ground_z", 0.0)
        self.declare_parameter("zone_lift", 0.04)
        self.declare_parameter("person_size", [0.6, 0.6, 1.8])
        self.declare_parameter("label_size", 0.55)       # m (world-size text)
        self.declare_parameter("show_status", True)
        self.declare_parameter("object_lifetime", 0.5)   # s
        # Unlabeled lidar clusters further away (houses, bale stacks) keep
        # their box but no text, to keep the view readable.
        self.declare_parameter("unknown_label_max_distance", 25.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.ground_z = float(g("ground_z"))
        self.zone_z = self.ground_z + float(g("zone_lift"))
        self.person_size = [float(v) for v in g("person_size")]
        self.label_size = float(g("label_size"))
        self.show_status = bool(g("show_status"))
        self.unknown_label_dist = float(g("unknown_label_max_distance"))
        lt = float(g("object_lifetime"))
        self.lifetime = Duration(sec=int(lt), nanosec=int((lt % 1) * 1e9))

        self.zones = []          # [(name, x_min, x_max, y_min, y_max)] red first
        self.zone_frame = "base_link"

        self.pub = self.create_publisher(SceneUpdate, g("scene_topic"), 10)
        latched = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL, depth=1)
        self.create_subscription(ZoneArray, g("zone_config_topic"), self._zones_cb, latched)
        self.create_subscription(TrackedObjectArray, g("detections_topic"), self._det_cb, 10)

    # ------------------------------------------------------------------

    def _zones_cb(self, msg: ZoneArray):
        order = {"red": 0, "yellow": 1}
        self.zones = sorted(((z.name, z.x_min, z.x_max, z.y_min, z.y_max) for z in msg.zones),
                            key=lambda z: order.get(z[0], 9))
        self.zone_frame = msg.header.frame_id or self.zone_frame
        self.get_logger().info(f"zones: {[z[0] for z in self.zones]} in {self.zone_frame}")

    def _zone_of(self, x, y):
        for name, x0, x1, y0, y1 in self.zones:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return None

    def _det_cb(self, msg: TrackedObjectArray):
        stamp = msg.header.stamp
        frame = msg.header.frame_id or self.zone_frame
        update = SceneUpdate()

        occupied = set()
        objects = SceneEntity(timestamp=stamp, frame_id=frame, id="objects",
                              lifetime=self.lifetime, frame_locked=True)
        for o in msg.objects:
            zone = self._zone_of(o.position.x, o.position.y)
            if zone:
                occupied.add(zone)
            self._add_object(objects, o, zone)
        update.entities.append(objects)

        # zones in every update: recorded scenes stay correct when seeking
        update.entities.append(self._zone_entity(stamp, occupied))

        status = SceneEntity(timestamp=stamp, frame_id=frame, id="status",
                             lifetime=self.lifetime, frame_locked=True)
        if self.show_status and "red" in occupied:
            n = sum(1 for o in msg.objects
                    if self._zone_of(o.position.x, o.position.y) == "red")
            status.texts.append(TextPrimitive(
                pose=_pose(0.0, 0.0, self.ground_z + 4.5), billboard=True,
                font_size=0.9, scale_invariant=False, color=_color(ZONE_RGB["red"], 1.0),
                text=f"STOP - {n} object{'s' if n != 1 else ''} in red zone"))
        update.entities.append(status)
        self.pub.publish(update)

    # ------------------------------------------------------------------

    def _add_object(self, ent: SceneEntity, o, zone):
        cls = o.class_name or "unknown"
        camera_only = o.source == "camera"
        if cls == "person" or camera_only:
            sx, sy, sz = self.person_size
            if not camera_only and o.size.z > 0.3:
                sx, sy, sz = max(o.size.x, sx), max(o.size.y, sy), max(o.size.z, 0.5)
        else:
            sx, sy, sz = max(o.size.x, 0.2), max(o.size.y, 0.2), max(o.size.z, 0.2)
        cx, cy = o.position.x, o.position.y
        # Camera-only objects are ground points; lidar objects are box centres.
        cz = self.ground_z + sz / 2 if camera_only else o.position.z

        rgb = ZONE_RGB[zone] if zone else CLASS_RGB.get(cls, DEFAULT_RGB)
        fill_a, edge_a = (0.12, 0.6) if camera_only else (0.3, 1.0)

        ent.cubes.append(CubePrimitive(pose=_pose(cx, cy, cz),
                                       size=_vec(sx, sy, sz), color=_color(rgb, fill_a)))
        ent.lines.append(LinePrimitive(type=LinePrimitive.LINE_LIST, pose=_pose(),
                                       thickness=0.04, scale_invariant=False,
                                       points=_box_edges(cx, cy, cz, sx, sy, sz),
                                       color=_color(rgb, edge_a)))
        dist = math.hypot(cx, cy)
        if cls == "unknown" and not zone and dist > self.unknown_label_dist:
            return
        tid = f" #{o.track_id}" if o.track_id >= 0 else ""
        tag = " (cam)" if camera_only else ""
        ent.texts.append(TextPrimitive(
            pose=_pose(cx, cy, cz + sz / 2 + 0.45), billboard=True,
            font_size=self.label_size, scale_invariant=False, color=_color(rgb, 1.0),
            text=f"{cls}{tid}{tag} {dist:.1f} m"))

    def _zone_entity(self, stamp, occupied):
        ent = SceneEntity(timestamp=stamp, frame_id=self.zone_frame, id="zones",
                          lifetime=self.lifetime, frame_locked=True)
        z = self.zone_z
        # draw the outer (yellow) zone first so the red one lies on top
        for i, (name, x0, x1, y0, y1) in enumerate(reversed(self.zones)):
            rgb = ZONE_RGB.get(name, (0.8, 0.8, 0.8))
            active = name in occupied
            zz = z + 0.01 * i
            corners = [Point(x=x0, y=y0, z=zz), Point(x=x1, y=y0, z=zz),
                       Point(x=x1, y=y1, z=zz), Point(x=x0, y=y1, z=zz)]
            ent.triangles.append(TriangleListPrimitive(
                pose=_pose(), points=[corners[k] for k in (0, 1, 2, 0, 2, 3)],
                color=_color(rgb, 0.28 if active else 0.08)))
            ent.lines.append(LinePrimitive(
                type=LinePrimitive.LINE_LOOP, pose=_pose(), thickness=0.12 if active else 0.07,
                scale_invariant=False, points=corners, color=_color(rgb, 1.0 if active else 0.7)))
            ent.texts.append(TextPrimitive(
                pose=_pose((x0 + x1) / 2, y1 - 0.8, zz + 0.05), billboard=False,
                font_size=0.9, scale_invariant=False, color=_color(rgb, 0.9),
                text=ZONE_LABEL.get(name, name.upper())))
        return ent


def _vec(x, y, z):
    return Vector3(x=float(x), y=float(y), z=float(z))


def main(args=None):
    rclpy.init(args=args)
    node = SafetyZoneSceneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
