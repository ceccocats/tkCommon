# check for driveworks
set(DriveWorks_FOUND False)
if( IS_DIRECTORY $ENV{HOME}/nvidia/nvidia_sdk/DRIVE_Software_9.0_Linux_hyperion_E3550/DriveSDK/drive-t186ref-linux/include/)
    set(DriveWorks_DIR $ENV{HOME}/nvidia/nvidia_sdk/DRIVE_Software_9.0_Linux_hyperion_E3550/DriveSDK/drive-t186ref-linux/include/)
    set(DriveWorks_FOUND True)
elseif( IS_DIRECTORY $ENV{HOME}/nvidia/nvidia_sdk/DRIVE_Software_10.0_Linux_OS_DDPX/DRIVEOS/drive-t186ref-linux/include/)
    set(DriveWorks_DIR $ENV{HOME}/nvidia/nvidia_sdk/DRIVE_Software_10.0_Linux_OS_DDPX/DRIVEOS/drive-t186ref-linux/include/)
    set(DriveWorks_FOUND True)
elseif( IS_DIRECTORY $ENV{HOME}/drive-t186ref-linux/include/)
    set(DriveWorks_DIR $ENV{HOME}/drive-t186ref-linux/include/)
    set(DriveWorks_FOUND True)
endif()

if(DriveWorks_FOUND)
    if(IS_DIRECTORY /usr/local/driveworks/)
        # get driwework path
        execute_process(
                COMMAND realpath /usr/local/driveworks/
                OUTPUT_VARIABLE DW_PATH
        )

        string(STRIP ${DW_PATH} DW_PATH)
        message("-- Found Driveworks: ${DW_PATH}")
        #set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR})
        add_definitions(-DVIBRANTE)
        #set(DriveWorks_FOUND True)
        set(DriveWorks_INCLUDE_DIRS
            /usr/local/cuda/targets/${CMAKE_SYSTEM_PROCESSOR}-linux/include/
            #/usr/local/driveworks/include/
            #/usr/local/driveworks/samples/src/
            #/usr/local/cuda/targets/aarch64-linux/include/
            #$ENV{HOME}/drive-t186ref-linux/include/
            #/usr/local/cuda/lib64/
            ${DriveWorks_DIR}
            /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/include
        )
        set(DriveWorks_LIBRARIES
            /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/lib/libdriveworks.so
            /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/lib/libdriveworks_visualization.so
            /usr/local/driveworks/targets/${CMAKE_SYSTEM_PROCESSOR}-Linux/lib/libcudnn.so.7
            EGL
        )

        # get DriveWork version
        get_directory_property(MYDEFS COMPILE_DEFINITIONS)
        if((MYDEFS MATCHES "^DW_VERSION_MAJOR=" OR MYDEFS MATCHES ";DW_VERSION_MAJOR=") AND
           (MYDEFS MATCHES "^DW_VERSION_MINOR=" OR MYDEFS MATCHES ";DW_VERSION_MINOR="))
            MESSAGE("-- DW_VERSION defined")
        else()
            # You can define your OS here if desired
            if (DW_PATH MATCHES ".*2\.2.*")
                set(DW_VERSION_MAJOR 2)
                set(DW_VERSION_MINOR 2)
                message("-- Driveworks VERSION: 2.2")
            elseif(DW_PATH MATCHES ".*2\.0.*")
                set(DW_VERSION_MAJOR 2)
                set(DW_VERSION_MINOR 0)
                message("-- Driveworks VERSION: 2.0")
            else()
                message(FATAL_ERROR "Driveworks VERISON not supported")
            endif()
        endif()
    else()
        set(DriveWorks_FOUND False)
    endif()
endif()


if(NOT DriveWorks_FOUND)
	message(WARNING "-- Driveworks NOT FOUND")
endif()


