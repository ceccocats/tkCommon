CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(tkCommon)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} --std=c++11 -fPIC -fopenmp")
set(CMAKE_BUILD_TYPE "Release")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#-------------------------------------------------------------------------------
# Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Pangolin REQUIRED)
include_directories(
    ${Pangolin_INCLUDE_DIRS}
)

set(GUI_LIBS
    ${Pangolin_LIBRARIES}
)

#-------------------------------------------------------------------------------
# library
#-------------------------------------------------------------------------------
add_library(tkGUI SHARED src/gui/Viewer.cpp)
target_link_libraries(tkGUI ${GUI_LIBS})

add_library(tkCommon SHARED src/common.cpp)
target_link_libraries(tkCommon tkGUI yaml-cpp)

#-------------------------------------------------------------------------------
# apps
#-------------------------------------------------------------------------------
add_executable(tkGUI_simple app/tkGUI_simple.cpp)
target_link_libraries(tkGUI_simple tkCommon ${GUI_LIBS})

#-------------------------------------------------------------------------------
# install
#-------------------------------------------------------------------------------
#install(TARGETS tkGUI DESTINATION lib)
#install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" # source directory
#        DESTINATION "include/"
#)
#install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
#        DESTINATION "share/tkCommon/cmake/" # target directory
#)



