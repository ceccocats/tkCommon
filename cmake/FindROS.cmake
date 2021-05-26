# find roscpp
find_package(roscpp QUIET)
find_package(rosbag QUIET)

if (roscpp_FOUND)
    set(ROS_INCLUDE_DIRS ${roscpp_INCLUDE_DIRS} ${rosbag_INCLUDE_DIRS})
    set(ROS_LIBRARIES ${roscpp_LIBRARIES} ${rosbag_LIBRARIES})
    message("-- Found ROS") #${ROS_INCLUDE_DIRS} 
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -DROS_ENABLED")
    set(ROS_FOUND true)
else()  
    message("-- ROS NOT FOUND")
    set(ROS_FOUND false)
endif() 