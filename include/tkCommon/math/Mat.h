
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

//#define STATIC  0
//#define DYNAMIC 1

namespace tk { namespace math {

/**
 * @brief   Matrix in cuda. Matrix in COLS order.
 * @tparam  T matrix class type
 */
template<class T>
class Mat : public tk::math::MatDump {

    protected:
        static const int MAXSIZE_MARGIN = 0;
        int     _maxSize = 0;
        int     _rows = 0;
        int     _cols = 0;
        int     _size = 0;
        bool    _gpu  = false;

    public:
        T*      data_d;
        T*      data_h;

        __host__
        Mat(){
            data_d = nullptr;
            data_h = nullptr;
        }

        __host__
        ~Mat(){
            if(data_h != nullptr){
                delete [] data_h;
                data_h = nullptr;
            } 
            if(data_d != nullptr && _gpu == true){
                HANDLE_ERROR( cudaFree(data_d) );
                data_d = nullptr;
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

            if(_maxSize < r * c){
                _maxSize = r * c + MAXSIZE_MARGIN;
                if(data_h != nullptr)
                    delete [] data_h;
                data_h = new T[_maxSize];
            }

            if(_gpu == true){
                if(_maxSize < r * c) {
                    if(data_d != nullptr)
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
        operator()(int r, int c) {
            tkASSERT(r < this->_rows && r >= 0, "Out of memory, rows "+std::to_string(r)+" > "+std::to_string(this->_rows)+"\n")
            tkASSERT(c < this->_cols && c >= 0, "Out of memory, cols "+std::to_string(c)+" > "+std::to_string(this->_cols)+"\n")
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
                (*this)(i/_cols, i%_cols) = a;
                i++;
            }

            if(_gpu == true){
                synchGPU();
            }
        }

        __host__ 
        Mat<T>& operator=(const Mat<T>& s) {
            cloneCPU(s);
            if(s.hasGPU() == true)
                cloneGPU(s);
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const Mat& s) {
            os<<"Mat ("<<s.rows()<<"x"<<s.cols()<<")";
            return os;
        }

        void print() {
            for(int i = 0; i < rows(); ++i) {
                std::cout << std::endl;
                for(int j = 0; j < rows(); ++j) {
                    std::cout << std::right << std::setw(20) << data_h[i+j*rows()];
                }
            }
            std::cout<<std::endl;
        }
};

template<class T, int R, int C>
class MatStatic : public tk::math::Mat<T> {
private:
        T staticdata_h[R*C];

public:
    MatStatic() : Mat<T>() {
        
        this->data_h = staticdata_h;
        this->_maxSize = R*C;
        this->_rows    = R;
        this->_cols    = C;
        this->_size    = R*C;
    }

    ~MatStatic() {
        this->data_h = nullptr;
    }

    __host__ void 
    resize(int r, int c) {
        tkASSERT(r==R && c==C, "unable to resize a static Mat");
    }

    __host__ 
    MatStatic<T,R,C>& operator=(const Mat<T>& s) {
        this->cloneCPU(s);
        if(s.hasGPU() == true)
            this->cloneGPU(s);
        return *this;
    }

};

typedef MatStatic<double,2,2> Mat2d;
typedef MatStatic<double,3,3> Mat3d;
typedef MatStatic<double,4,4> Mat4d;

typedef MatStatic<int,2,2> Mat2i;
typedef MatStatic<int,3,3> Mat3i;
typedef MatStatic<int,4,4> Mat4i;

typedef MatStatic<float,2,2> Mat2f; 
typedef MatStatic<float,3,3> Mat3f; 
typedef MatStatic<float,4,4> Mat4f; 

}}