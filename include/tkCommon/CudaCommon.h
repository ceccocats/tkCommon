#pragma once
//#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

static void inline HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s:%d:0\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

namespace tk{
    namespace cuda{

/**
 * @brief   Class for implement a matrix in cuda. Matrix in ROW order.
 * 
 * @tparam  T matrix class type
 */
template<class T>
class tkMatrixCuda{

    private:

        int     max_size;

    public:

        int     rows;
        int     cols;
        int     size;
        T*      data;


        //host function
        //////////////////////////////////////////////////////////////////////////////////////////
        __host__ bool 
        init(int r, int c){

            rows        = r;
            cols        = c;
            size        = r * c;
            max_size    = r * c + 10000;  //TODO

            HANDLE_ERROR( cudaMalloc(&data, max_size * sizeof(T)) );

            return true;
        }

        __host__ void
        fromDevice(T* p){ HANDLE_ERROR( cudaMemcpy(p, data, size * sizeof(T), cudaMemcpyDeviceToHost) ); }

        __host__ void
        fromHost(T* p){ HANDLE_ERROR( cudaMemcpy(data, p, size * sizeof(T), cudaMemcpyHostToDevice) ); }

        __host__ void
        memset(T v){ HANDLE_ERROR( cudaMemset(data,v, size * sizeof(T)) );}

        __host__ void
        resize(int r, int c){
            if(max_size < r * c){
                max_size = r * c + 10000;
                HANDLE_ERROR( cudaFree(data) );
                HANDLE_ERROR( cudaMalloc(&data, max_size * sizeof(T)) );
            }

            rows = r;
            cols = c;
            size = r * c;
        }

        __host__ void
        close(){
            HANDLE_ERROR( cudaFree(data) ); 
        }

        __device__ T&
        at(int r, int c){ return data[r+c*rows]; }

};


}}