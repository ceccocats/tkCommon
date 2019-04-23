CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(tkCommon)

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")
set(CMAKE_BUILD_TYPE "Release")
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(QGLViewer REQUIRED)
find_package(Qt5 COMPONENTS Core Xml OpenGL Gui Widgets)
find_package(OpenGL REQUIRED)
set(QT_USE_QTOPENGL TRUE)
set(QT_USE_QTXML TRUE)
include_directories(
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

#-------------------------------------------------------------------------------
# library
#-------------------------------------------------------------------------------
add_library(tkGUI SHARED src/gui/lodepng.cpp src/gui/Viewer.cpp)
target_link_libraries(tkGUI ${GUI_LIBS})

add_library(tkCommon SHARED src/common.cpp)
target_link_libraries(tkCommon tkGUI)

# QT5 AUTOMOC
set_target_properties(tkGUI PROPERTIES AUTOMOC TRUE)

#-------------------------------------------------------------------------------
# apps
#-------------------------------------------------------------------------------
add_executable(tkGUI_simple app/tkGUI_simple.cpp)
target_link_libraries(tkGUI_simple tkCommon ${GUI_LIBS})


qt5_wrap_cpp(UI_CPP app/Viewer2dgui.h)
QT5_WRAP_UI(UI_HEADERS app/Viewer2dgui.ui)
add_executable(tkGUI_2dgui app/tkGUI_2dgui.cpp ${UI_HEADERS} ${UI_CPP})
target_link_libraries(tkGUI_2dgui tkCommon ${GUI_LIBS})


#-------------------------------------------------------------------------------
# install
#-------------------------------------------------------------------------------
install(TARGETS tkGUI DESTINATION lib)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/include/" # source directory
        DESTINATION "include/"
)
install(DIRECTORY "${CMAKE_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkGUI/cmake/" # target directory
)



