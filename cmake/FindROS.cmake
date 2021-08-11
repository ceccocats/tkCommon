# FindROS
include(CMakeFindDependencyMacro)

set(ROS_VERSION $ENV{ROS_VERSION})

if (ROS_VERSION EQUAL 1)
    find_dependency(roscpp)
    find_dependency(rosbag)
    find_dependency(tf)
elseif (ROS_VERSION EQUAL 2)
    find_dependency(rclcpp)
    find_dependency(ament_cmake)
    #find_dependency(rosidl_typesupport_cpp)
endif()

find_dependency(sensor_msgs)
find_dependency(nav_msgs)
find_dependency(geometry_msgs)
find_dependency(ackermann_msgs)
find_dependency(tf2_msgs)

if (ROS_VERSION EQUAL 1 
    AND roscpp_FOUND 
    AND rosbag_FOUND 
    AND tf_FOUND 

    AND sensor_msgs_FOUND 
    AND nav_msgs_FOUND 
    AND tf2_msgs_FOUND 
    AND ackermann_msgs_FOUND
    AND geometry_msgs_FOUND
)
    set(ROS_INCLUDE_DIRS 
        ${roscpp_INCLUDE_DIRS} 
        ${rosbag_INCLUDE_DIRS} 
        ${tf_INCLUDE_DIRS}
        
        ${sensor_msgs_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
        ${geometry_msgs_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
        ${ackermann_msgs_INCLUDE_DIRS}
        )
    set(ROS_LIBRARIES 
        ${roscpp_LIBRARIES} 
        ${rosbag_LIBRARIES} 
        ${tf_LIBRARIES}
        
        ${sensor_msgs_LIBRARIES}
        ${nav_msgs_LIBRARIES}
        ${geometry_msgs_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
        ${ackermann_msgs_LIBRARIES} 
    )
    message("-- Found ROS")
    set(ROS_FOUND true)
    
elseif (ROS_VERSION EQUAL 2
    AND rclcpp_FOUND 
    AND ament_cmake_FOUND
    #AND rosidl_typesupport_cpp_FOUND
    
    AND sensor_msgs_FOUND 
    AND nav_msgs_FOUND 
    AND tf2_msgs_FOUND 
    AND ackermann_msgs_FOUND
    AND geometry_msgs_FOUND 
)
    set(ROS_INCLUDE_DIRS
        ${rclcpp_INCLUDE_DIRS}
        ${ament_cmake_INCLUDE_DIRS}
        #${rosidl_typesupport_cpp_INCLUDE_DIRS}
        
        ${sensor_msgs_INCLUDE_DIRS}
        ${nav_msgs_INCLUDE_DIRS}
        ${geometry_msgs_INCLUDE_DIRS}
        ${tf2_msgs_INCLUDE_DIRS}
        ${ackermann_msgs_INCLUDE_DIRS}
    )
    set(ROS_LIBRARIES
        ${rclcpp_LIBRARIES}
        ${ament_cmake_LIBRARIES}
        #${rosidl_typesupport_cpp_LIBRARIES}

        ${sensor_msgs_LIBRARIES}
        ${nav_msgs_LIBRARIES}
        ${geometry_msgs_LIBRARIES}
        ${tf2_msgs_LIBRARIES}
        ${ackermann_msgs_LIBRARIES} 
    )
    message("-- Found ROS2")
    set(ROS_FOUND true)
else()  
    message("-- ROS NOT FOUND")
    set(ROS_FOUND false)
endif()
