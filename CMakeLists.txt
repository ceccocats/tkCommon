cmake_minimum_required(VERSION 3.18...3.19)

project(tkCommon VERSION 1.0
                 DESCRIPTION "tkCommon project"
                 LANGUAGES CXX)

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ standard to use")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/tkConf.cmake OPTIONAL)

#-------------------------------------------------------------------------------
# CUDA
#-------------------------------------------------------------------------------
find_package(CUDAToolkit QUIET)

#-------------------------------------------------------------------------------
# External Libraries
#-------------------------------------------------------------------------------
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw3 3 REQUIRED)
find_package(Freetype REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(ROS QUIET)
find_package(ROS2 QUIET)

#-------------------------------------------------------------------------------
# Generate Data Headers
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/GpsData_gen.h
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/_genDataHeaders.py
    COMMAND python3 _genDataHeaders.py)
add_custom_target(tkdataheaders_gen DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/data/gen/GpsData_gen.h)

#-------------------------------------------------------------------------------
# Generate shaders colorsmaps
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/colormaps/*
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/glsl/pointcloudFrag
    COMMAND python3 _colormapsGen.py)
add_custom_target(tkshaderpointcloud_gen DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/shader/colormaps/*)

#-------------------------------------------------------------------------------
# Generate Drawables include
#-------------------------------------------------------------------------------
add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/drawables/*
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/drawables
    COMMAND python3 _drawables.py)
add_custom_target(tkdrawablesgen DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/tkCommon/gui/drawables/*)

#-------------------------------------------------------------------------------
# tkCommon library
#-------------------------------------------------------------------------------
file(GLOB data_gen  include/tkCommon/data/gen/*.cpp)
file(GLOB data      include/tkCommon/data/*.cpp)
add_library(common SHARED
    src/common.cpp
    src/log.cpp
    src/version.cpp
    src/CmdParser.cpp 
    src/rt/Profiler.cpp
    src/PID.cpp
    ${data_gen}
    ${data} 
)
target_link_libraries(common 
    OpenMP::OpenMP_CXX 
    Eigen3::Eigen
    ${YAML_CPP_LIBRARIES}
    matio 
    dl 
    ${ROS_LIBRARIES}
    ${ROS2_LIBRARIES}
)
target_include_directories(common PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
    ${ROS_INCLUDE_DIRS}
    ${ROS2_INCLUDE_DIRS}
)
if (CUDAToolkit_FOUND)
    target_link_libraries(common CUDA::cudart CUDA::cuda_driver)
    target_compile_options(common PUBLIC -DTKCUDA_ENABLED)
    set_property(TARGET common PROPERTY CUDA_SEPARABLE_COMPILATION ON) 
endif()
if(ROS_FOUND)
    target_compile_options(common PUBLIC -DROS_ENABLED)
elseif(ROS2_FOUND)
    target_compile_options(common PUBLIC -DROS2_ENABLED)
endif()
add_library(${PROJECT_NAME}::common ALIAS common)

#-------------------------------------------------------------------------------
# tkGUI library
#-------------------------------------------------------------------------------
file(GLOB gui_utils src/gui/utils/*.cpp)
file(GLOB imgui     src/gui/imgui/*.cpp)
file(GLOB implot    src/gui/implot/*.cpp)
file(GLOB drawables src/gui/drawables/*.cpp)
file(GLOB shader    src/gui/shader/*.cpp)

add_library(gui SHARED
    src/gui/Viewer.cpp
    ${drawables}
    ${gui_utils}  
    ${imgui}
    ${implot}
    ${shader}
)
target_link_libraries(gui 
    OpenGL::GL
    GLUT::GLUT
    Freetype::Freetype
    GLEW::GLEW
    glfw
    #Xxf86vm
    #X11
    #gcc_s
    #gcc 
    tkCommon::common
)
add_library(${PROJECT_NAME}::gui ALIAS gui)

#-------------------------------------------------------------------------------
# tkMath library
#-------------------------------------------------------------------------------
add_library(math SHARED 
    src/math/CSpline.cpp 
    src/math/CSpline2D.cpp 
    src/math/CSpline3D.cpp
)
target_link_libraries(math 
    tkCommon::common
)
add_library(${PROJECT_NAME}::math ALIAS math)

#-------------------------------------------------------------------------------
# tkCommunication library
#-------------------------------------------------------------------------------
add_library(communication SHARED
    src/communication/ethernet/UDPSocket.cpp
    src/communication/ethernet/TCPSocket.cpp
    src/communication/ethernet/PCAPHandler.cpp
    src/communication/ethernet/PacketParser.cpp
    src/communication/EthInterface.cpp
    src/communication/CanInterface.cpp
    src/communication/serial/SerialPort.cpp
    src/communication/car/CarControl.cpp
)
target_link_libraries(communication tkCommon::common pcap serial)
add_library(${PROJECT_NAME}::communication ALIAS communication)

#-------------------------------------------------------------------------------
# tkJoystick library
#-------------------------------------------------------------------------------
add_library(joystick SHARED 
    src/joystick/Joystick.cpp
    src/joystick/gamepad/gamepad.cpp
)
target_link_libraries(joystick udev)
target_include_directories(joystick PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include> 
)
add_library(${PROJECT_NAME}::joystick ALIAS joystick)

#-------------------------------------------------------------------------------
# tkSensor library
#-------------------------------------------------------------------------------
add_library(sensor SHARED
    src/sensor/Sensor.cpp
    src/sensor/SensorManager.cpp
    src/sensor/LogManager.cpp    
)
target_link_libraries(sensor
    tkCommon::common tkCommon::gui tkCommon::communication tkCommon::math 
)
add_library(${PROJECT_NAME}::sensor ALIAS sensor)

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
add_dependencies(common tkversiongit tkdataheaders_gen)
add_dependencies(gui tkversiongit tkdataheaders_gen tkshaderpointcloud_gen tkdrawablesgen)

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
    target_link_libraries(${app} tkCommon::common tkCommon::gui tkCommon::communication tkCommon::math tkCommon::sensor tkCommon::joystick)
endforeach()

#-------------------------------------------------------------------------------
# Install
#-------------------------------------------------------------------------------
install(TARGETS ${tkpackage_APPS} 
    DESTINATION "bin/"
)
install(TARGETS common gui communication math joystick sensor
    DESTINATION "lib/"
    EXPORT ${PROJECT_NAME}Targets
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/scripts/" 
    DESTINATION "bin/"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/" 
    DESTINATION "include/"
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/cmake/" 
    DESTINATION "share/${PROJECT_NAME}/cmake/" 
)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/data/" 
    DESTINATION "share/${PROJECT_NAME}/data"
)

#-------------------------------------------------------------------------------
# Export Configuration
#-------------------------------------------------------------------------------
install(EXPORT ${PROJECT_NAME}Targets NAMESPACE ${PROJECT_NAME}::
    FILE ${PROJECT_NAME}Targets.cmake
    DESTINATION "share/${PROJECT_NAME}/cmake/"
)

include(tkExportConfig)
export_config(${PROJECT_NAME})

# install the configuration file
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION "share/${PROJECT_NAME}/cmake/"
)
