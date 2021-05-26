#pragma once
#include <tkCommon/log.h>

#ifdef TKCUDA_ENABLED
    #include <cuda.h>
    #include <cuda_runtime.h>
    static void inline tkCudaHandleError( cudaError_t err,
                            const char *file,
                            int line ) {
        if (err != cudaSuccess) {
            tkERR(cudaGetErrorString( err )<<"\n");
            tkERR("file: "<<file<<":"<<line<<"\n");
            throw std::runtime_error("cudaError");
        }
    }
    #define tkCUDA( err ) (tkCudaHandleError( err, __FILE__, __LINE__ ))

#else

#define tkCUDA(x)  (tkWRN("compiled without cuda, this feature is not avaible\n"))
#define __host__ 
#define __device__
#endif
