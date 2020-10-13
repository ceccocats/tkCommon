CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(tkCommon)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tkConf.cmake OPTIONAL)

# project specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDA 9.0 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIR})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
find_package(glfw3 3 REQUIRED)
find_library(GLFW3_LIBRARY NAMES glfw3 glfw)

find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})

set(GUI_LIBS
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${GLEW_LIBRARIES}
    ${GLFW3_LIBRARY}
    glut
    ${FREETYPE_LIBRARIES}
    Xxf86vm
    X11
    gcc_s
    gcc
    ${CUDA_LIBRARIES}
)


#-------------------------------------------------------------------------------
# Generate Data Headers
#-------------------------------------------------------------------------------
#add_custom_target(tkdataheaders_gen 
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen
#    COMMAND python3 _genDataHeaders.py)

# definition of fidl_list
add_custom_command(OUTPUT 
        ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/GpsData_gen.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/_genDataHeaders.py
    COMMAND python3 _genDataHeaders.py)
add_custom_target(tkdataheaders_gen DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/GpsData_gen.h)

add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/colormaps/*
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/glsl/pointcloudFrag
    COMMAND python3 _colormapsGen.py)
    add_custom_target(tkshaderpointcloud_gen DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/colormaps/*)

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
add_library(tkGUI SHARED
    src/gui/Viewer.cpp
    src/gui/utils/PlotManager.cpp
    src/gui/utils/Buffer.cpp
    src/gui/utils/Shader.cpp
    src/gui/utils/Texture.cpp
    src/gui/utils/Camera.cpp
    src/gui/utils/stb_image.cpp
    src/gui/utils/CommonViewer.cpp
    src/gui/imgui/imgui_demo.cpp
    src/gui/imgui/imgui_draw.cpp
    src/gui/imgui/imgui_widgets.cpp
    src/gui/imgui/imgui.cpp
    src/gui/imgui/imgui_impl_glfw.cpp
    src/gui/imgui/imgui_impl_opengl3.cpp
    src/common.cpp
    src/version.cpp
)
target_link_libraries(tkGUI ${GUI_LIBS})

add_library(tkCommon SHARED
    src/common.cpp
    src/version.cpp
    src/CmdParser.cpp    
)
target_link_libraries(tkCommon tkGUI yaml-cpp matio dl)

add_library(tkCommunication SHARED
    src/communication/ethernet/UDPSocket.cpp
    src/communication/ethernet/TCPSocket.cpp
    src/communication/ethernet/PCAPHandler.cpp
    src/communication/ethernet/PacketParser.cpp
    src/communication/EthInterface.cpp
    src/communication/CanInterface.cpp
    src/communication/serial/SerialPort.cpp
    src/communication/car/CarControl.cpp
)
target_link_libraries(tkCommunication pcap)

add_library(tkMath SHARED src/math/CSpline.cpp src/math/CSpline2D.cpp src/math/CSpline3D.cpp)

add_library(tkJoystick SHARED 
    src/joystick/Joystick.cpp
    src/joystick/gamepad/gamepad.cpp
)
target_link_libraries(tkJoystick udev)

#-------------------------------------------------------------------------------
# GIT version autoupdate
#-------------------------------------------------------------------------------
include_directories(${CMAKE_CURRENT_BINARY_DIR}) # to find the file on build dir

set(defineCMD "\\\#define TKVERSION_GIT \\\"") 
execute_process(COMMAND "git status > /dev/null" RESULT_VARIABLE ISNT_GIT_REPO)
if( NOT ${ISNT_GIT_REPO})
    message("-- Found GIT")
    # write git version to file
    add_custom_target(tkversiongit_tmp COMMAND echo ${defineCMD}`git log --format="%h" -n 1`"\\\"" > tkversion_git.tmp)
else()
    message("-- GIT: THIS IS NOT A REPO, VERSIONING IS DISABLED")
    add_custom_target(tkversiongit_tmp COMMAND echo ${defineCMD}"000000\\\"" > tkversion_git.tmp)
endif()

# copy only if it is changed (this solves useless recompiles)
add_custom_target(tkversiongit COMMAND rsync --checksum tkversion_git.tmp tkversion_git.h)
add_dependencies(tkversiongit tkversiongit_tmp)
add_dependencies(tkCommon tkversiongit tkdataheaders_gen)
add_dependencies(tkGUI tkversiongit tkdataheaders_gen tkshaderpointcloud_gen)


#-------------------------------------------------------------------------------
# Build apps
#-------------------------------------------------------------------------------
file(GLOB tkpackage_APPScpp "app/*.cpp")
foreach(cpp ${tkpackage_APPScpp})
    get_filename_component(app "${cpp}" NAME_WE)
    list(APPEND tkpackage_APPS ${app})
endforeach()
foreach(app ${tkpackage_APPS})
    add_executable(${app} "app/${app}.cpp")
    target_link_libraries(${app} tkCommon tkCommunication tkJoystick ${GUI_LIBS})
endforeach()


#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
#install(TARGETS ${tkpackage_APPS} DESTINATION bin )
#install(TARGETS tkCommon tkGUI tkMath tklibDrawText tkCommunication DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/" # source directory
        DESTINATION "bin/"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" # source directory
        DESTINATION "include/"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" # source directory
        DESTINATION "share/tkCommon/cmake/" # target directory
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/" # source directory
        DESTINATION "share/tkCommon/data"
)



