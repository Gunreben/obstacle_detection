# obstacle_detection

Lidar obstacle clustering + camera/lidar late fusion + safety zones for
the vario700 tractor demonstrator (ROS 2 Humble).

## Pipeline

```
/ouster/points ─▶ lidar_filter (box/ground) ─▶ /ouster/points/filtered
                                                      │
                       obstacle_detection_node / _cuda│(clustering+tracking)
                                                      ▼
/yolo/<cam>/detections ──────────────▶ detection_fusion_node ◀── /lidar/objects
        (yolo_multi_cam)                     │
                                             ├─▶ /safety_zone/detections  (TrackedObjectArray)
                                             ├─▶ /safety_zone/alerts      (SafetyAlertArray, every cycle)
                                             └─▶ /safety_zone/config      (ZoneArray, latched)
```

Full stack launch (TF + lidar filter + clustering + YOLO + fusion):

```bash
ros2 launch obstacle_detection safety_zone_visualizer_stack.launch.py
```

## Nodes

### obstacle_detection_node (CPU) / obstacle_detection_cuda (GPU)

Euclidean clustering of a (pre-filtered) point cloud, with optional
near/mid/far adaptive parameters and persistent track IDs
(`cluster_tracker.hpp`, gated nearest-neighbor + EMA smoothing).

The CUDA node uses a source-built voxel connected-components kernel
(`src/cudaEuclideanCluster.cu`, replaces the prebuilt cuPCL binary in
`lib/` — that binary predates JetPack 6 and is kept for reference only).
Built for sm_75/80/86/87/89 (87 = Jetson AGX Orin).

Outputs:
- `/lidar/objects` (`TrackedObjectArray`) — **the machine-readable
  interface** consumed by the fusion node (class `unknown`, source
  `lidar`).
- `/obstacle_markers` (`MarkerArray`) — debug visualization
  (`publish_markers`, default true).
- `/detected_obstacles` (colored cluster `PointCloud2`) — debug only,
  costs a full cloud republish per frame (`publish_cluster_cloud`,
  disable on the Jetson).

See `config/params.yaml` / `config/params_cuda.yaml` for tuning
(cluster sizes, adaptive ranges, tracking gates).

### detection_fusion_node.py

Late fusion of YOLO 2D detections with lidar cluster objects, running
once per `/lidar/objects` message (~10 Hz):

1. **Pixel association:** lidar object centers are projected into each
   camera *including lens distortion* (the ArkCams are kannala_brandt
   fisheye; the ZED plumb_bob/rational) and matched against YOLO bboxes
   (+`bbox_margin_pixels`); the closest object in range wins (occlusion).
2. **3D fallback:** unmatched detections are ground-projected
   (bbox bottom-center ray ∩ z=`ground_z`) and nearest-neighbor matched
   within `max_association_distance`.
3. **Camera-only:** remaining detections become `source: "camera"`
   objects at the ground position, deduplicated across overlapping
   cameras (`camera_only_merge_distance`).

Camera detections expire after `detection_timeout` (default 0.7 s) and
`yolo_multi_cam` publishes empty arrays per processed frame, so objects
vanish promptly when no longer seen (v0.1 kept the last detection of a
camera alive forever — ghost objects).

The camera optical frame is taken from the `Detection2DArray` header,
and v1/v2 camera_info topics can both be configured
(`camera_info_topic_v1/_v2`) — the node works against v1 bags and the
live v2 stack without reconfiguration. For v1 bags, run
`ros2 launch tractor_multi_cam_publisher camera_info_only.launch.py`
alongside (v1 bags contain no camera_info).

`/safety_zone/alerts` is published every cycle (empty when clear) so
visualizers can clear their alert state. `/safety_zone/config` is
latched (transient_local) zone geometry for the Foxglove
SafetyZoneVisualizer panel.

**Zone convention:** zones are currently interpreted as x = lateral,
y = forward, matching the mesh-derived TF tree. Verifying/fixing this
against REP-103 is parked as a separate HIL step — do not change it as
a side effect of other work.

## Running the demonstrator on the tractor (D4-validated)

```bash
# 1. Sensors + TF (recording configuration)
ros2 launch tractor_bringup tractor_bringup.launch.py
# 2. (optional) dataset recording
bash /mnt/ros2_ws/system/scripts/record.sh
# 3. Detection stack — bringup owns TF, hence tf:=false
ros2 launch obstacle_detection safety_zone_visualizer_stack.launch.py tf:=false
# 4. Visualization for the Foxglove panel
ros2 run foxglove_bridge foxglove_bridge   # connect Foxglove to ws://<jetson>:8765
```

Measured on the AGX Orin (2026-07-07, all sensors + recording +
detection concurrently, 120 s):

| Metric | Value |
|---|---|
| Sensor rates | Ouster 10.0 Hz, ArkCams 24.6 Hz, ZED 14.7 Hz (all nominal) |
| Detection rates | /lidar/objects 10.0 Hz, fusion + alerts 10.0 Hz, YOLO ~8 Hz/cam |
| Alert latency (lidar stamp → alert) | **280 ms** (152 ms of that is inside the Ouster driver) |
| CPU (detection + recording) | filter 0.64 + cluster 0.73 + yolo 1.13 + fusion 0.15 + recorder 0.47 ≈ 3.1 cores; system loadavg 8.1/12 |
| GPU | ~52% (yolo TensorRT + ZED NEURAL depth) |
| Recording | ~70 MB/s to disk, zero recorder drops, bag counts nominal |

Detection topics (`/lidar/objects`, `/safety_zone/*`, `/yolo/*`) are
**not** in the dataset recording list; add them to
`tractor_bringup/config/record_topics.txt` if a run should capture them.

## Messages

`TrackedObject(Array)`, `SafetyAlert(Array)`, `Zone(Array)` — see
`msg/`.

## Requirements

ROS 2 Humble, PCL (`ros-humble-pcl-ros`), CUDA toolkit (CUDA node),
`ros-humble-vision-msgs` + Python `opencv` + `numpy` (fusion node),
`lidar_filter` package (https://github.com/Gunreben/ROS2-Pointcloud-Filter-Tools),
`yolo_multi_cam` for the camera branch.

```bash
colcon build --packages-select obstacle_detection
```

## Changelog

### 0.2.0 (2026-07)
- Cluster nodes publish `TrackedObjectArray` on `/lidar/objects`;
  markers and the colored cluster cloud demoted to optional debug
  outputs (`publish_markers`, `publish_cluster_cloud`).
- Fusion rewritten: consumes `/lidar/objects` instead of re-processing
  the raw point cloud per YOLO callback (was O(131k points × cameras ×
  frame rate) in Python — the main obstacle to Jetson realtime);
  distortion-aware projection (fisheye ArkCams — fixes duplicate
  detections at image edges); stale-detection expiry (fixes ghost
  objects); cross-camera dedup; alerts published every cycle;
  latched `/safety_zone/config`.
- YOLO input topics moved to `yolo_multi_cam` 0.2.0's stable
  `/yolo/<name>/detections` names; camera frames from detection
  headers (v1 bag + v2 live compatible).
- CUDA arch list extended with sm_87 (Jetson AGX Orin);
  `safety_zone_visualizer_stack.launch.py` now actually loads
  `params_cuda.yaml`.
- Merged `desktop-optimization` branch into `main`.

### 0.1.x
- Desktop demonstrator: CPU/CUDA clustering, tracking, marker-based
  fusion, zone alerts.
