set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_SYSROOT $ENV{HOME}/drive-t186ref-linux/targetfs)

set(CMAKE_C_COMPILER /usr/bin/aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER /usr/bin/aarch64-linux-gnu-g++)

set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# This set the target for 'nvcc' that used by CUDA_TOOLKIT
#set(CUDA_TARGET_CPU_ARCH ${CMAKE_SYSTEM_PROCESSOR})
set(CUDA_TARGET_OS_VARIANT "linux")
set(cuda_target_full_path ${CMAKE_SYSTEM_PROCESSOR}-${CUDA_TARGET_OS_VARIANT})
set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda)
set(CUDA_TOOLKIT_INCLUDE ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/include)
set(CUDA_CUDART_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/libcudart.so)
set(CUDA_cublas_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libcublas.so)
set(CUDA_cufft_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libcufft.so)
set(CUDA_nppc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppc.so)
set(CUDA_nppial_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppial.so)
set(CUDA_nppicc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppicc.so)
set(CUDA_nppicom_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppicom.so)
set(CUDA_nppidei_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppidei.so)
set(CUDA_nppif_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppif.so)
set(CUDA_nppig_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppig.so)
set(CUDA_nppim_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppim.so)
set(CUDA_nppist_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppist.so)
set(CUDA_nppisu_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppisu.so)
set(CUDA_nppitc_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnppitc.so)
set(CUDA_npps_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libnpps.so)
set(CUDA_cusolver_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/stubs/libcusolver.so)
set(CUDA_cudadevrt_LIBRARY ${CUDA_TOOLKIT_ROOT_DIR}/targets/${cuda_target_full_path}/lib/libcudadevrt.a)

include_directories(/usr/include/aarch64-linux-gnu/)
include_directories(${CMAKE_SYSROOT}/usr/local/include)


# copy all libraries to install dir
if(NOT TARGET install-deps)
    add_custom_target(install-deps
        COMMAND echo "install external libraries"
        COMMAND mkdir -p ${CMAKE_INSTALL_PREFIX}/libEXT/
        COMMAND rsync -avh --ignore-errors ${CMAKE_SYSROOT}/usr/local/lib/ ${CMAKE_INSTALL_PREFIX}/libEXT/
        COMMAND rsync -avh --ignore-errors ${CMAKE_SYSROOT}/usr/lib/ ${CMAKE_INSTALL_PREFIX}/libEXT/
        COMMAND rsync -avh --ignore-errors ${CMAKE_SYSROOT}/opt/pdk/lib/ ${CMAKE_INSTALL_PREFIX}/libEXT/
        COMMAND rsync -avh --ignore-errors ${CMAKE_SYSROOT}/opt/ros/*/lib/ ${CMAKE_INSTALL_PREFIX}/libEXT/
    )
endif()

if(NOT TARGET upload)
    # set UPLOAD to target
    set(TK_USER nvidia)
    set(TK_PASS nvidia)
    #set(TK_IP 192.168.1.207)
    set(TK_TARGET_INSTALL_PATH /home/${TK_USER}/build)
    add_custom_target(upload
        # create installation folder on target
        COMMAND sshpass -p "${TK_PASS}" ssh -o StrictHostKeyChecking=no ${TK_USER}@${TK_IP} "mkdir -p ${TK_TARGET_INSTALL_PATH}"
        # upload installation
        COMMAND sshpass -p "${TK_PASS}" rsync --progress -rltgDz -e "ssh" ${CMAKE_INSTALL_PREFIX}/ ${TK_USER}@${TK_IP}:${TK_TARGET_INSTALL_PATH}/
        COMMAND echo "installed to ${TK_IP}:${TK_TARGET_INSTALL_PATH}"
    )
endif()