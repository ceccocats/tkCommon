# FindROS
include(CMakeFindDependencyMacro)
if ($ENV{ROS_VERSION} EQUAL 2)
    find_dependency(rclcpp)
    find_dependency(ament_cmake)
    #find_dependency(rosbag)
    find_dependency(sensor_msgs)
    find_dependency(nav_msgs)
    find_dependency(tf2_msgs)
    find_dependency(rosidl_typesupport_cpp)
endif()

if (rclcpp_FOUND AND
    sensor_msgs_FOUND AND
    nav_msgs_FOUND AND
    tf2_msgs_FOUND AND
    ament_cmake_FOUND AND
    rosidl_typesupport_cpp_FOUND
)
    set(ROS2_INCLUDE_DIRS
        ${rclcpp_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
        ${ament_cmake_INCLUDE_DIRS}
        ${rosidl_typesupport_cpp_INCLUDE_DIRS}
    )
    set(ROS2_LIBRARIES
        ${roscpp_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${nav_msgs_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
        ${ament_cmake_LIBRARIES}
        ${rosidl_typesupport_cpp_LIBRARIES}
    )
    message("-- Found ROS2")
    set(ROS2_FOUND true)
else()  
    #set(ROS2_INCLUDE_DIRS "")
    #set(ROS2_LIBRARIES "")
    message("-- ROS2 NOT FOUND")
    set(ROS2_FOUND false)
endif() 
