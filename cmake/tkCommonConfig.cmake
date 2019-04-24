set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")

find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

# compile GUI only if QGLViewer is installed
find_package(QGLViewer REQUIRED)
find_package(Qt5 COMPONENTS Core Xml OpenGL Gui Widgets)
find_package(OpenGL REQUIRED)
set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTXML TRUE)
set(GUI_INCLUDES
    ${Qt5Core_INCLUDE_DIRS} 
    ${Qt5Xml_INCLUDE_DIRS}
    ${Qt5OpenGL_INCLUDE_DIRS} 
    ${Qt5Gui_INCLUDE_DIRS} 
    ${Qt5Widgets_INCLUDE_DIRS} 
    ${QGLVIEWER_INCLUDE_DIR}
)
set(GUI_LIBS
    ${Qt5Core_LIBRARIES} 
    ${Qt5Xml_LIBRARIES}
    ${Qt5OpenGL_LIBRARIES} 
    ${Qt5Gui_LIBRARIES} 
    ${Qt5Widgets_LIBRARIES} 
    ${QGLVIEWER_LIBRARY}
    ${OPENGL_LIBRARIES}
)

set(tkCommon_INCLUDE_DIRS
    ${EIGEN3_INCLUDE_DIR}
    ${GUI_INCLUDES}
)

set(tkCommon_LIBRARIES 
    yaml-cpp
    ${GUI_LIBS}
    tkGUI
    #tkCommon
)

# This causes catkin_simple to link against these libraries
set(tkCommon_FOUND_CATKIN_PROJECT true)
