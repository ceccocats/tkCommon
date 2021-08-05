# FindROS
include(CMakeFindDependencyMacro)

set(ROS_VERSION $ENV{ROS_VERSION})

if (ROS_VERSION EQUAL 1)
    find_dependency(roscpp)
    find_dependency(rosbag)
elseif (ROS_VERSION EQUAL 2)
    find_dependency(rclcpp)
    find_dependency(ament_cmake)
    #find_dependency(rosidl_typesupport_cpp)
endif()

if (roscpp_FOUND AND
    rosbag_FOUND
)
    set(ROS_TEMP_INCLUDE_DIRS
        ${roscpp_INCLUDE_DIRS}
        ${rosbag_INCLUDE_DIRS}
    )
    set(ROS_TEMP_LIBRARIES
        ${roscpp_LIBRARIES}
        ${rosbag_LIBRARIES}
    )
    message("-- Found ROS")
    set(ROS_FOUND true)
    
elseif (rclcpp_FOUND AND
    ament_cmake_FOUND# AND
    #rosidl_typesupport_cpp_FOUND
)
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
    message("-- Found ROS2")
    set(ROS_FOUND true)
endif()

if (ROS_FOUND)
    find_dependency(sensor_msgs)
    find_dependency(nav_msgs)
    find_dependency(tf2_msgs)
    find_dependency(ackermann_msgs)
    if (sensor_msgs_FOUND AND
        nav_msgs_FOUND AND
        tf2_msgs_FOUND AND
        ackermann_msgs_FOUND
    )
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
    else()
        message("-- ROS MISSING DEPENDENCIES")
        set(ROS_FOUND false)
    endif()
    #unset(ROS_TEMP_INCLUDE_DIRS)
    #unset(ROS_TEMP_LIBRARIES)
else()  
    message("-- ROS NOT FOUND")
    set(ROS_FOUND false)
endif()
