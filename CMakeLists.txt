cmake_minimum_required(VERSION 3.8)
project(obstacle_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(CUDA REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}   # Add this line
  /usr/local/include
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# CPU-based obstacle detection node
add_executable(obstacle_detection_node src/obstacle_detection_node.cpp)
ament_target_dependencies(obstacle_detection_node
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  PCL
)
target_link_libraries(obstacle_detection_node ${PCL_LIBRARIES})

# CUDA-based obstacle detection node

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)


add_executable(obstacle_detection_cuda src/obstacle_detection_cuda.cpp)
ament_target_dependencies(obstacle_detection_cuda
  rclcpp
  sensor_msgs
  pcl_conversions
  pcl_ros
  visualization_msgs
  tf2
  tf2_ros
  tf2_sensor_msgs
  PCL
)
target_link_libraries(obstacle_detection_cuda ${PCL_LIBRARIES} ${CUDA_LIBRARIES} cudacluster)

# Install targets
install(TARGETS obstacle_detection_node obstacle_detection_cuda
  DESTINATION lib/${PROJECT_NAME})
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/lib/libcudacluster.so
  DESTINATION lib
)

# Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_cpplint_FOUND TRUE)
  set(ament_cmake_copyright_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

