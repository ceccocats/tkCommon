message("-- Found tkCommon")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# compile GUI only if QGLViewer is installed
find_package(Pangolin REQUIRED)
include_directories(
    ${Pangolin_INCLUDE_DIRS}
)

set(GUI_LIBS
    ${Pangolin_LIBRARIES}
)

set(tkCommon_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${GUI_INCLUDES}
)

set(tkCommon_LIBRARIES 
    yaml-cpp
    ${GUI_LIBS}
    tkGUI
)

# This causes catkin_simple to link against these libraries
set(tkCommon_FOUND true)
