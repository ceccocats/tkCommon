CMAKE_MINIMUM_REQUIRED(VERSION 3.5)

PROJECT(tkCommon)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tkConf.cmake OPTIONAL)

# project specific flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")


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

find_package(Freetype REQUIRED)
include_directories(${FREETYPE_INCLUDE_DIRS})

set(GUI_LIBS
    ${OPENGL_gl_LIBRARY}
    ${OPENGL_glu_LIBRARY}
    ${GLEW_LIBRARY}  
    glfw
    glut
    ${FREETYPE_LIBRARIES}
)


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
    src/gui/lodepng.cpp
    src/gui/MouseView3D.cpp
    src/gui/MathUtils.cpp
    src/gui/imgui_demo.cpp
    src/gui/imgui_draw.cpp
    src/gui/imgui_widgets.cpp
    src/gui/imgui.cpp
    src/gui/imgui_impl_glfw.cpp
    src/gui/imgui_impl_opengl3.cpp
    src/common.cpp
)
target_link_libraries(tkGUI tklibDrawText ${GUI_LIBS})

add_library(tkCommon SHARED
    src/common.cpp
    src/CmdParser.cpp
)
target_link_libraries(tkCommon tkGUI yaml-cpp matio dl)

add_library(tkCommunication SHARED
    src/communication/ethernet/UDPSocket.cpp
    src/communication/ethernet/PCAPHandler.cpp
    src/communication/EthInterface.cpp
)
target_link_libraries(tkCommunication pcap)

add_library(tkMath SHARED src/math/CSpline.cpp src/math/CSpline2D.cpp src/math/CSpline3D.cpp)

#-------------------------------------------------------------------------------
# Build apps
#-------------------------------------------------------------------------------
add_executable(tkGUI_simple app/tkGUI_simple.cpp)
target_link_libraries(tkGUI_simple tkCommon ${GUI_LIBS})

add_executable(tkCommon_cmd app/tkCommon_cmd.cpp)
target_link_libraries(tkCommon_cmd tkCommon)

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

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
install(TARGETS tkGUI_simple tkCommon_pcap_replay tkCommon_pcap_gui_replay tkCommon_eth_socket tkCommon_eth_pcap DESTINATION bin )
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



