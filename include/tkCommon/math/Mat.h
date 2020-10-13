
#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <initializer_list>
#include "tkCommon/CudaCommon.h"
#include "tkCommon/math/MatIO.h"

namespace tk { namespace math {

/**
 * @brief   Matrix in cuda. Matrix in COLS order.
 * @tparam  T matrix class type
 */
template<class T>
class Mat : public tk::math::MatDump {

    private:
        static const int MAXSIZE_MARGIN = 0;
        int     _maxSize = 0;
        int     _rows = 0;
        int     _cols = 0;
        int     _size = 0;
        bool    _gpu  = false;

    public:
        T*      data_d = nullptr;
        T*      data_h = nullptr;

        __host__
        Mat(){

        }

        __host__
        ~Mat(){
            if(data_h != nullptr){
                HANDLE_ERROR( cudaFreeHost(data_h) );
            } 
            if(data_d != nullptr && _gpu == true){
                HANDLE_ERROR( cudaFree(data_d) );
            } 
        }

        __host__ 
        Mat(int r, int c){
            if(r > 0 && c > 0){
                resize(r,c);
            }
        }

        __host__ void
        useGPU(){
            _gpu = true;
            resize(_rows,_cols);
        }

        __host__ bool
        hasGPU() const {
            return _gpu;
        }

        __host__ void 
        copyFrom(T*data, int r, int c) {
            resize(r, c);
            memcpy(data_h, data, _size*sizeof(T));
            if(_gpu == true){
                synchGPU();
            }
        }

        __host__ void 
        copyTo(T*data) {
            memcpy(data, data_h, _size*sizeof(T)); 
        }

        __host__ void 
        cloneGPU(const Mat<T> &m) {

            tkASSERT(m.hasGPU() == true, "Could not clone from GPU")

            resize(m.rows(), m.cols());
            useGPU();
            HANDLE_ERROR( cudaMemcpy(data_d, m.data_d, m.size() * sizeof(T), cudaMemcpyDeviceToDevice) ); 
        }

        __host__ void 
        cloneCPU(const Mat<T> &m) {

            resize(m.rows(), m.cols());
            memcpy(data_h, m.data_h, m.size() * sizeof(T)); 
        }
        
        __host__ void 
        synchGPU(){ 

            useGPU();
            HANDLE_ERROR( cudaMemcpy(data_d, data_h, _size * sizeof(T), cudaMemcpyHostToDevice) ); 
        }

        __host__ void 
        synchCPU(){ 

            tkASSERT(_gpu == true,"You set mat only on CPU\n");
            HANDLE_ERROR( cudaMemcpy(data_h, data_d, _size * sizeof(T), cudaMemcpyDeviceToHost) ); 
        }
        
        __host__ void 
        resize(int r, int c) {

            if(_maxSize < r * c || data_h == nullptr){
                _maxSize = r * c + MAXSIZE_MARGIN;
                HANDLE_ERROR( cudaFreeHost(data_h) );
                HANDLE_ERROR( cudaMallocHost(&data_h, _maxSize * sizeof(T)) );
            }

            if(_gpu == true){
                if(_maxSize < r * c || data_d == nullptr){
                    HANDLE_ERROR( cudaFree(data_d) );
                    HANDLE_ERROR( cudaMalloc(&data_d, _maxSize * sizeof(T)) );
                }
            }
            _rows = r;
            _cols = c;
            _size = r * c;
        }


        __host__ __device__ int 
        rows() const { 
            return _rows; 
        }

        __host__ __device__ int 
        cols() const { 
            return _cols; 
        }
        
        __host__ __device__ int 
        size() const { 
            return _size; 
        }
        
        __host__ T&  
        atCPU(int r, int c) { 
            return data_h[r+c*_rows]; 
        }
        
        __device__ T&  
        atGPU(int r, int c) { 
            if(_gpu == true){
                return data_d[r+c*_rows];
            }
            return nullptr;
        }

        __host__ void 
        set(std::initializer_list<T> a_args) {
            if(a_args.size() != size()) {
                std::cout<<"ERROR: you must set all data\n";
                std::cout<<"try insert: "<<a_args.size()<<" in size: "<<size()<<"\n";
                exit(1);
            }
            int i=0;
            for(auto a : a_args) {
                atCPU(i/_cols, i%_cols) = a;
                i++;
            }

            if(_gpu == true){
                synchGPU();
            }
        }

        __host__ 
        Mat& operator=(const Mat& s) {
            cloneCPU(s);
            cloneGPU(s);
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const Mat& s) {
            std::cout<<"Mat ("<<s.rows()<<"x"<<s.cols()<<")";
            for(int i = 0; i < s.rows(); ++i) {
                std::cout << std::endl;
                for(int j = 0; j < s.rows(); ++j) {
                    std::cout << std::right << std::setw(20) << s.data_h[i+j*s.rows()];
                }
            }
            return os;
        }
};

class Mat3d : public Mat<double> {
    Mat3d() {
        resize(3,3);
    }
};

class Mat4f : public Mat<float> {
    Mat4f() {
        resize(4,4);
    }
};

}}