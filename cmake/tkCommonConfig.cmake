message("-- Found tkCommon")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")

find_package(Eigen3 REQUIRED)
#find_package(yaml-cpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3 REQUIRED)
find_library(GLFW3_LIBRARY NAMES glfw3 glfw)

find_package(Freetype REQUIRED)

set(tkCommon_INCLUDE_DIRS
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${FREETYPE_INCLUDE_DIRS}
)

set(tkCommon_LIBRARIES 
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
    ${FREETYPE_LIBRARIES}
    tklibDrawText
    matio
)

set(tkCommon_FOUND true)
