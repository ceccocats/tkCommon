#include "tkCommon/catch2/catch.hpp"
#include "tkCommon/common.h"
#include "tkCommon/math/Mat.h"
#include "tkCommon/math/MatIO.h"
#include "tkCommon/math/Vec.h"
#include <cuda.h>
#include "tkCommon/rt/Profiler.h"

TEST_CASE("Test cuda class") {
    volatile float *matNew = new float[1000000];
    volatile float *matMHost;
    cudaMallocHost(&matMHost, 1000000 * sizeof(float));
    volatile float *matMDevice;
    cudaMalloc(&matMDevice, 1000000 * sizeof(float));
    volatile float *matManaged;
    cudaMallocManaged(&matManaged, 1000000 * sizeof(float));

    SECTION("New to gpu") {
        int i = 0;
        for(i = 0; i < 1000; i++)
        {
            memset((void*)matNew,i,1000000*sizeof(float));
            tkPROF_tic(New_to_GPU);
            cudaMemcpy((void*)matMDevice, (void*)matNew, 1000000*sizeof(float), cudaMemcpyHostToDevice);
            tkPROF_toc(New_to_GPU);

            memset((void*)matMHost,i,1000000*sizeof(float));
            tkPROF_tic(Host_to_GPU);
            cudaMemcpy((void*)matMDevice, (void*)matMHost, 1000000*sizeof(float), cudaMemcpyHostToDevice);
            tkPROF_toc(Host_to_GPU);

            cudaMemset((void*)matMDevice,i,1000000*sizeof(float));
            tkPROF_tic(GPU_to_New);
            cudaMemcpy((void*)matNew, (void*)matMDevice, 1000000*sizeof(float), cudaMemcpyDeviceToHost);
            tkPROF_toc(GPU_to_New);

            cudaMemset((void*)matMDevice,i,1000000*sizeof(float));
            tkPROF_tic(GPU_to_Host);
            cudaMemcpy((void*)matMHost, (void*)matMDevice, 1000000*sizeof(float), cudaMemcpyDeviceToHost);
            tkPROF_toc(GPU_to_Host);

            memset((void*)matManaged,i,1000000*sizeof(float));
            tkPROF_tic(Managed_to_GPU);
            cudaMemcpy((void*)matMDevice, (void*)matManaged, 1000000*sizeof(float), cudaMemcpyHostToDevice);
            tkPROF_toc(Managed_to_GPU);

            cudaMemset((void*)matManaged,i,1000000*sizeof(float));
            tkPROF_tic(Managed_to_New);
            cudaMemcpy((void*)matNew, (void*)matManaged, 1000000*sizeof(float), cudaMemcpyDeviceToHost);
            tkPROF_toc(Managed_to_New);

            cudaMemset((void*)matManaged,i,1000000*sizeof(float));
            tkPROF_tic(Managed_to_Host);
            cudaMemcpy((void*)matMHost, (void*)matManaged, 1000000*sizeof(float), cudaMemcpyDeviceToHost);
            tkPROF_toc(Managed_to_Host);
        }
    }

    SECTION("Print"){
        tkPROF_print
    }
}