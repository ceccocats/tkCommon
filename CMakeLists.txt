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
message(${GUI_LIBS})


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

#-------------------------------------------------------------------------------
# Build Libraries
#-------------------------------------------------------------------------------
add_library(tklibDrawText SHARED
    src/gui/libdrawtext/draw.c
    src/gui/libdrawtext/drawgl.c
    src/gui/libdrawtext/drawrast.c
    src/gui/libdrawtext/font.c
    src/gui/libdrawtext/tpool.c
    src/gui/libdrawtext/utf8.c
)

add_library(tkGUI SHARED
    src/gui/Viewer.cpp
    src/gui/PlotManager.cpp
    src/gui/Buffer.cpp
    src/gui/Shader.cpp
    src/gui/Texture.cpp
    src/gui/Camera.cpp
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
target_link_libraries(tkGUI tklibDrawText ${GUI_LIBS})

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
)
target_link_libraries(tkCommunication pcap)

add_library(tkMath SHARED src/math/CSpline.cpp src/math/CSpline2D.cpp src/math/CSpline3D.cpp)


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
add_dependencies(tkGUI tkversiongit tkdataheaders_gen)


#-------------------------------------------------------------------------------
# Build apps
#-------------------------------------------------------------------------------
add_executable(tkGUI_new app/tkGUI_new.cpp)
target_link_libraries(tkGUI_new tkCommon)

add_executable(tkCommon_cmd app/tkCommon_cmd.cpp)
target_link_libraries(tkCommon_cmd tkCommon)

add_executable(tkCommon_matio app/tkCommon_matio.cpp)
target_link_libraries(tkCommon_matio tkCommon)

add_executable(tkCommon_eth_pcap app/tkCommon_eth_pcap.cpp)
target_link_libraries(tkCommon_eth_pcap tkCommon tkCommunication)

add_executable(tkCommon_pcap_replay app/tkCommon_pcap_replay.cpp)
target_link_libraries(tkCommon_pcap_replay tkCommon tkCommunication)

add_executable(tkCommon_eth_socket app/tkCommon_eth_socket.cpp)
target_link_libraries(tkCommon_eth_socket tkCommon tkCommunication)

add_executable(tkCommon_terminal_format app/tkCommon_terminal_format.cpp)
target_link_libraries(tkCommon_terminal_format)

add_executable(tkCommon_pcap_gui_replay app/tkCommon_pcap_gui_replay.cpp)
target_link_libraries(tkCommon_pcap_gui_replay tkCommon tkCommunication ${GUI_LIBS})

add_executable(tkCommon_segfault app/tkCommon_segfault.cpp)
target_link_libraries(tkCommon_segfault tkCommon)

add_executable(tkCommon_geoconv app/tkCommon_geoconv.cpp)
target_link_libraries(tkCommon_geoconv tkCommon)

add_executable(tkCommon_can app/tkCommon_can.cpp)
target_link_libraries(tkCommon_can tkCommon tkCommunication)

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
install(TARGETS tkGUI_new tkCommon_pcap_replay tkCommon_pcap_gui_replay tkCommon_eth_socket tkCommon_eth_pcap DESTINATION bin )
install(TARGETS tkCommon tkGUI tkMath tklibDrawText tkCommunication DESTINATION lib)
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



