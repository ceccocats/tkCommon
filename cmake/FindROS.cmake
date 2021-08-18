# FindROS
include(CMakeFindDependencyMacro)

set(ROS_VERSION $ENV{ROS_VERSION})
set(ROS_FOUND false)

if (ROS_VERSION EQUAL 1)
    message("-- Search ROS1")
    find_dependency(roscpp)
    find_dependency(rosbag)
    set(ROS_TEMP_INCLUDE_DIRS
        ${roscpp_INCLUDE_DIRS}
        ${rosbag_INCLUDE_DIRS}
    )
    set(ROS_TEMP_LIBRARIES
        ${roscpp_LIBRARIES}
        ${rosbag_LIBRARIES}
    )
elseif (ROS_VERSION EQUAL 2)
    message("-- Search ROS2")
    set(AMENT_CMAKE_UNINSTALL_TARGET OFF) # solve multiple uninstall target issue on dashing
    find_dependency(rclcpp)
    find_dependency(ament_cmake)
    #find_dependency(rosidl_typesupport_cpp)
    set(ROS_TEMP_INCLUDE_DIRS
        ${rclcpp_INCLUDE_DIRS}
        ${ament_cmake_INCLUDE_DIRS}
        #${rosidl_typesupport_cpp_INCLUDE_DIRS}
    )
    set(ROS_TEMP_LIBRARIES
        ${rclcpp_LIBRARIES}
        ${ament_cmake_LIBRARIES}
        #${rosidl_typesupport_cpp_LIBRARIES}
    )
else()
    message("-- ROS not FOUND")
    set(ROS_FOUND false)
endif()

if (ROS_VERSION EQUAL 1 OR ROS_VERSION EQUAL 2)

    find_dependency(sensor_msgs)
    find_dependency(nav_msgs)
    find_dependency(tf2_msgs)
    find_dependency(ackermann_msgs)
    set(ROS_INCLUDE_DIRS
        ${ROS_TEMP_INCLUDE_DIRS}
        ${sensor_msgs_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
        ${ackermann_msgs_INCLUDE_DIRS}
    )
    set(ROS_LIBRARIES
        ${ROS_TEMP_LIBRARIES}
        ${sensor_msgs_LIBRARIES}
        ${nav_msgs_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
        ${ackermann_msgs_LIBRARIES}
    )
    list(REMOVE_DUPLICATES ROS_LIBRARIES) # faster compile

    message("-- ROS FOUND")
    set(ROS_FOUND true)

    unset(ROS_TEMP_INCLUDE_DIRS)
    unset(ROS_TEMP_LIBRARIES)
endif()