message("-- Found tkCommon")
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3 REQUIRED)

set(GUI_INCLUDES
    ${OPENGL_INCLUDE_DIR}
    ${GLEW_INCLUDE_DIRS}
)

set(GUI_LIBS
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${GLEW_LIBRARY}    
    glfw
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

set(tkCommon_FOUND true)
