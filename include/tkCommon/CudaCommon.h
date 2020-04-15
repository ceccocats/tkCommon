#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

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
        int*    d_rows;
        int*    d_cols;

        int     h_rows;
        int     h_cols;

        int     max_size;
    
    public:
        T*      data;


        //host function
        //////////////////////////////////////////////////////////////////////////////////////////
        __host__ bool 
        init(int rows, int cols){

            h_rows      = rows;
            h_cols      = cols;
            max_size    = rows * cols;

            HANDLE_ERROR( cudaMalloc(&data, max_size * sizeof(T)) );
            HANDLE_ERROR( cudaMalloc(&d_rows, sizeof(int)) );
            HANDLE_ERROR( cudaMalloc(&d_cols, sizeof(int)) );

            HANDLE_ERROR( cudaMemcpy(d_rows, &rows, sizeof(int), cudaMemcpyHostToDevice) );
            HANDLE_ERROR( cudaMemcpy(d_cols, &cols, sizeof(int), cudaMemcpyHostToDevice) );

            return true;
        }

        __host__ void
        DeviceToHostMemcpy(T* p){ HANDLE_ERROR( cudaMemcpy(p, data, hostSize() * sizeof(T), cudaMemcpyDeviceToHost) ); }

        __host__ void
        HostToDeviceMemcpy(T* p){ HANDLE_ERROR( cudaMemcpy(data, p, hostSize() * sizeof(T), cudaMemcpyHostToDevice) ); }

        __host__ int
        hostCols(){ return h_cols; }

        __host__ int
        hostRows(){ return h_rows; }

        __host__ int
        hostSize(){ return h_cols*h_rows; }

        __host__ void
        hostResize(int rows, int cols){
            if(max_size < rows*cols){
                max_size = rows*cols+10000;
                HANDLE_ERROR( cudaFree(data) );
                HANDLE_ERROR( cudaMalloc(&data, max_size * sizeof(T)) );
            }

            HANDLE_ERROR( cudaMemcpy(d_rows, &rows, sizeof(int), cudaMemcpyHostToDevice) );
            HANDLE_ERROR( cudaMemcpy(d_cols, &cols, sizeof(int), cudaMemcpyHostToDevice) );
            h_rows = rows;
            h_cols = cols;
        }

        __host__ void
        close(){
            HANDLE_ERROR( cudaFree(data) ); 
            HANDLE_ERROR( cudaFree(d_rows) ); 
            HANDLE_ERROR( cudaFree(d_cols) ); 
        }

        //device function
        //////////////////////////////////////////////////////////////////////////////////////////

        __device__ int
        deviceCols(){ return *d_cols; }

        __device__ int
        deviceRows(){ return *d_rows; }

        __device__ int
        deviceSize(){ return *(d_cols)*(*d_rows); }

        __device__ T
        at(int r, int c){ return data[r+c*(*d_rows)]; }

        __device__ void
        set(int r, int c,T value ){ data[r+c*(*d_rows)] = value; }
};


/**
 * @brief   Class for implement a pinned matrix. Matrix in ROW order.
 * 
 * @tparam  T matrix class type
 */
template<class T>
class tkPinnedMatrix{

    public:
        int     h_rows;
        int     h_cols;

        int     max_size;
    
    public:
        T*      data;

        __host__ bool 
        init(int rows, int cols){

            h_rows      = rows;
            h_cols      = cols;
            max_size    = rows * cols;

            HANDLE_ERROR( cudaMallocHost(&data, max_size * sizeof(T)) );

            return true;
        }

        __host__ void
        DeviceToHostMemcpy(T* p){ HANDLE_ERROR( cudaMemcpy(p, data, size() * sizeof(T), cudaMemcpyDeviceToHost) ); }

        __host__ void
        HostToDeviceMemcpy(T* p){ HANDLE_ERROR( cudaMemcpy(data, p, size() * sizeof(T), cudaMemcpyDeviceToHost) ); }

        __host__ int
        cols(){ return h_cols; }

        __host__ int
        rows(){ return h_rows; }

        __host__ int
        size(){ return h_cols*h_rows; }

        __host__ void
        resize(int rows, int cols){
            if(max_size < rows*cols){
                max_size = rows*cols+10000;
                HANDLE_ERROR( cudaFree(data) );
                HANDLE_ERROR( cudaMallocHost(&data, max_size * sizeof(T)) );
            }
            h_rows = rows;
            h_cols = cols;
        }

        __host__ T
        at(int r, int c){ return data[r+c*h_rows]; }

        __host__ void
        set(int r, int c,T value ){ data[r+c*h_rows] = value; }

        __host__ void
        close(){ /*HANDLE_ERROR( cudaFree(data) );*/ }

};

/**
 * @brief   Class for implement a unified matrix. Matrix in ROW order.
 * 
 * @tparam  T matrix class type
 */
template<class T>
class tkUnifiedMatrix{

    public:
        int*     u_rows;
        int*     u_cols;

        int*     u_max_size;
    
    public:
        T*      data;

        __host__ __device__ bool 
        init(int rows, int cols){

            HANDLE_ERROR( cudaMallocManaged(&u_rows, sizeof(int)) );
            HANDLE_ERROR( cudaMallocManaged(&u_cols, sizeof(int)) );
            HANDLE_ERROR( cudaMallocManaged(&u_max_size, sizeof(int)) );

            *u_rows      = rows;
            *u_cols      = cols;
            *u_max_size  = rows * cols;

            HANDLE_ERROR( cudaMallocManaged(data, (*u_max_size) * sizeof(T)) );

            return true;
        }

        __host__ __device__ int
        cols(){ return *u_cols; }

        __host__ __device__ int
        rows(){ return *u_rows; }

        __host__ __device__ int
        size(){ return (*u_cols)*(*u_rows); }

        __host__ __device__ void
        resize(int rows, int cols){
            if((*u_max_size) < rows*cols){
                *u_max_size  = rows * cols + 10000;
                HANDLE_ERROR( cudaFree(data) );
                HANDLE_ERROR( cudaMallocManaged(data, (*u_max_size) * sizeof(T)) );
            }

            *u_rows  = rows;
            *u_cols  = cols;
        }

        __host__ __device__ T
        at(int r, int c){ return data[r+c*(*u_rows)]; }

        __host__ __device__ void
        set(int r, int c,T value ){ data[r+c*(*u_rows)] = value; }

        __host__ __device__ void
        close(){ 
            HANDLE_ERROR( cudaFree(data) ); 
            HANDLE_ERROR( cudaFree(u_rows) ); 
            HANDLE_ERROR( cudaFree(u_cols) ); 
        }

};


}}