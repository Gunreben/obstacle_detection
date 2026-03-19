cmake_minimum_required(VERSION 3.18)
project(obstacle_detection LANGUAGES C CXX CUDA)

# ── Compiler options ─────────────────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Target RTX 4090 (Ada Lovelace, sm_89).
# Also include sm_75/80/86 for Turing / Ampere portability.
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA architectures")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD  17)

# ── ROS 2 dependencies ───────────────────────────────────────────────────────
find_package(ament_cmake        REQUIRED)
find_package(rclcpp             REQUIRED)
find_package(sensor_msgs        REQUIRED)
find_package(std_msgs           REQUIRED)
find_package(geometry_msgs      REQUIRED)
find_package(pcl_conversions    REQUIRED)
find_package(pcl_ros            REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2                REQUIRED)
find_package(tf2_ros            REQUIRED)
find_package(tf2_sensor_msgs    REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# ── Custom message generation ─────────────────────────────────────────────────
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TrackedObject.msg"
  "msg/TrackedObjectArray.msg"
  "msg/SafetyAlert.msg"
  "msg/SafetyAlertArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

# ── PCL (CPU modules always required) ───────────────────────────────────────
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)

# ── CUDA Toolkit (modern CMake target-based) ─────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── PCL GPU modules (optional – only available when PCL is built with CUDA) ──
find_package(PCL QUIET COMPONENTS gpu_segmentation gpu_octree gpu_filters)
if(PCL_FOUND
   AND TARGET pcl_gpu_segmentation
   AND TARGET pcl_gpu_octree
   AND TARGET pcl_gpu_filters)
  set(HAVE_PCL_GPU TRUE)
  message(STATUS "PCL GPU modules found – building obstacle_detection_gpu")
else()
  set(HAVE_PCL_GPU FALSE)
  message(STATUS "PCL GPU modules NOT found – skipping obstacle_detection_gpu")
endif()

# ── Common include / link dirs ───────────────────────────────────────────────
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# ============================================================================
# 1. CPU obstacle detection node
# ============================================================================
add_executable(obstacle_detection_node src/obstacle_detection_node.cpp)
ament_target_dependencies(obstacle_detection_node
  rclcpp sensor_msgs pcl_conversions pcl_ros
  visualization_msgs tf2 tf2_ros tf2_sensor_msgs PCL)
target_link_libraries(obstacle_detection_node ${PCL_LIBRARIES})

# ============================================================================
# 2. CUDA cluster library (source-built for this machine's architecture)
#    Replaces the precompiled ARM binary (lib/libcudacluster.so).
# ============================================================================
add_library(cudacluster SHARED src/cudaEuclideanCluster.cu)
target_include_directories(cudacluster PUBLIC
  include
  ${CUDAToolkit_INCLUDE_DIRS})
set_target_properties(cudacluster PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(cudacluster
  CUDA::cudart)

# ============================================================================
# 3. CUDA obstacle detection node
# ============================================================================
add_executable(obstacle_detection_cuda src/obstacle_detection_cuda.cpp)
ament_target_dependencies(obstacle_detection_cuda
  rclcpp sensor_msgs pcl_conversions pcl_ros
  visualization_msgs tf2 tf2_ros tf2_sensor_msgs PCL)
target_link_libraries(obstacle_detection_cuda
  ${PCL_LIBRARIES}
  cudacluster
  CUDA::cudart)

# ============================================================================
# 4. PCL-GPU obstacle detection node (optional)
# ============================================================================
if(HAVE_PCL_GPU)
  add_executable(obstacle_detection_gpu src/obstacle_detection_node_gpu.cpp)
  ament_target_dependencies(obstacle_detection_gpu
    rclcpp sensor_msgs pcl_conversions pcl_ros
    visualization_msgs tf2 tf2_ros tf2_sensor_msgs PCL)
  target_link_libraries(obstacle_detection_gpu
    ${PCL_LIBRARIES}
    pcl_gpu_segmentation
    pcl_gpu_octree
    pcl_gpu_filters)
endif()

# ============================================================================
# Install
# ============================================================================
set(INSTALL_TARGETS obstacle_detection_node obstacle_detection_cuda)
if(HAVE_PCL_GPU)
  list(APPEND INSTALL_TARGETS obstacle_detection_gpu)
endif()

install(TARGETS ${INSTALL_TARGETS} cudacluster
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

install(DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
  OPTIONAL)

install(PROGRAMS scripts/detection_fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
  OPTIONAL)

# ── Testing ──────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND  TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
