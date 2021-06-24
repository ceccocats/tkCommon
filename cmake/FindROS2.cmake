# FindROS
include(CMakeFindDependencyMacro)
if ($ENV{ROS_VERSION} EQUAL 2)
    find_dependency(rclcpp)
    #find_dependency(rosbag)
    find_dependency(sensor_msgs)
endif()

if (rclcpp_FOUND AND
    sensor_msgs_FOUND
)
    set(ROS2_INCLUDE_DIRS ${rclcpp_INCLUDE_DIRS} ${sensor_msgs_INCLUDE_DIRS})
    set(ROS2_LIBRARIES ${roscpp_LIBRARIES} ${sensor_msgs_LIBRARIES})
    message("-- Found ROS2")
    set(ROS2_FOUND true)
else()  
    #set(ROS2_INCLUDE_DIRS "")
    #set(ROS2_LIBRARIES "")
    message("-- ROS2 NOT FOUND")
    set(ROS2_FOUND false)
endif() 
