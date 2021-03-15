message("-- Found tkCommon")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")

find_package(CUDA 9.0 QUIET)
include_directories(${CUDA_INCLUDE_DIRS})
if(CUDA_FOUND)
    add_definitions(-DTKCUDA_ENABLED)
endif()

find_package(Eigen3 REQUIRED)
#find_package(yaml-cpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3 REQUIRED)
find_library(GLFW3_LIBRARY NAMES glfw3 glfw)
find_package(ROS QUIET)

find_package(Freetype REQUIRED)

set(tkCommon_INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIRS}
    ${ROS_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
)

set(tkCommon_LIBRARIES 
    ${CUDA_LIBRARIES}
    yaml-cpp
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARY}
    glut
    tkCommon
    tkGUI
    tkMath
    tkCommunication
    tkSensor
    ${FREETYPE_LIBRARIES}
    matio
    tkJoystick
    ${ROS_LIBRARIES}
)

set(tkCommon_FOUND true)
