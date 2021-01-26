
#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <initializer_list>
#include "tkCommon/cuda.h"
#include "tkCommon/math/MatIO.h"

namespace tk { namespace math {

// GPU INFO
static int  Mat_device = -1;
static bool Mat_integrated_memory = false;

template<class T>
struct Mat_t {
    T* data;
    int rows, cols;
    int size;
    int maxSize;

    bool integrated_memory;

    __host__
    Mat_t() {
        data = nullptr;
        rows = cols = size = maxSize = 0;
        integrated_memory = false;
    }

    __device__ T&  
    at(int r, int c) { 
        return data[r+c*rows];
    }

    __host__
    void resize(int r, int c) {
        if(maxSize < r * c) {
            maxSize = (r * c);
                
            if(data != nullptr)
                if(!integrated_memory)
                    tkCUDA( cudaFree(data) );
                else
                    tkCUDA( cudaFreeHost(data) );

            if(!integrated_memory)
                tkCUDA( cudaMalloc(&data, maxSize * sizeof(T)) );
            else
                tkCUDA( cudaMallocHost(&data, maxSize * sizeof(T)) );
        }
        rows = r;
        cols = c;        
    }

    __host__
    void release() {
        if(data != nullptr)
            tkCUDA( cudaFree(data) );
    }
    

};

/**
 * @brief   Matrix in cuda. Matrix in COLS order.
 * @tparam  T matrix class type
 */
template<class T, int R = -1, int C = -1>
class Mat : public Eigen::Matrix<T, R, C>, public tk::math::MatDump {

protected:
    bool    _use_gpu;

    void init() {
        _use_gpu = false;

#ifdef TKCUDA_ENABLED
        // get GPU type
        if(Mat_device == -1) {
            cudaDeviceProp  props;
            tkCUDA(cudaGetDevice(&Mat_device));
            tkCUDA(cudaGetDeviceProperties(&props,Mat_device));
            Mat_integrated_memory = props.integrated != 0;
        }
        gpu.integrated_memory = Mat_integrated_memory;
#endif
    }

public:
    Mat_t<T> gpu;

    Mat(void) : Eigen::Matrix<T, R, C>() {
        init();
    }


    template<typename OtherDerived>
    Mat(const Eigen::MatrixBase<OtherDerived>& other) : Eigen::Matrix<T, R, C>(other) { 
        init();
    }

    template<typename OtherDerived>
    Mat& operator=(const Eigen::MatrixBase <OtherDerived>& other) {
        this->Eigen::Matrix<T, R, C>::operator=(other);
        return *this;
    }

    __host__
    ~Mat() {
        gpu.release();
    }


    __host__ void
    useGPU(){
        _use_gpu = true;
        resize(Eigen::Matrix<T,R,C>::rows(),Eigen::Matrix<T,R,C>::cols());
    }

    __host__ bool
    hasGPU() const {
        return _use_gpu;
    }

    __host__ void 
    copyFrom(T*data, int r, int c) {
        resize(r, c);
        memcpy(Eigen::Matrix<T,R,C>::data(), data, Eigen::Matrix<T,R,C>::size()*sizeof(T));
        if(_use_gpu == true){
            synchGPU();
        }
    }

    __host__ void 
    copyTo(T*data) {
        memcpy(data, Eigen::Matrix<T,R,C>::data(), Eigen::Matrix<T,R,C>::size()*sizeof(T)); 
    }
    
    __host__ void 
    synchGPU(){ 
        useGPU();
        tkCUDA( cudaMemcpy(gpu.data, Eigen::Matrix<T,R,C>::data(), Eigen::Matrix<T,R,C>::size() * sizeof(T), cudaMemcpyHostToDevice) ); 
    }

    __host__ void 
    synchCPU(){ 
        tkASSERT(_use_gpu == true, "You set mat only on CPU\n");
        tkASSERT(Eigen::Matrix<T,R,C>::rows() == gpu.rows && Eigen::Matrix<T,R,C>::cols() == gpu.cols, "GPU dimensions dont match");
        tkCUDA( cudaMemcpy(Eigen::Matrix<T,R,C>::data(), gpu.data, Eigen::Matrix<T,R,C>::size() * sizeof(T), cudaMemcpyDeviceToHost) ); 
    }
    
    __host__ void 
    resize(int r, int c) {
        tkASSERT(R == -1 && C == -1, "You cant resize a static Mat");
        Eigen::Matrix<T,R,C>::resize(r,c);
        if(_use_gpu == true){
            gpu.resize(r, c);
        }
    }

    __host__ T&
    operator[](int n) {
        //tkASSERT(n < this->_size, "Out of memory " + std::to_string(n) + " of " + std::to_string(this->_size));
        return Eigen::Matrix<T,R,C>::data()[n]; 
    }

    __host__ Mat<T,R,C>& 
    operator=(const Mat<T,R,C>& s) {
        Eigen::Matrix<T,R,C> &m = *this;
        m = s;
        if(_use_gpu) {
            this->synchGPU();
        }
        return *this;
    }

    friend std::ostream& 
    operator<<(std::ostream& os, const Mat& s) {
        os<<"Mat ("<<s.rows()<<"x"<<s.cols()<<")";
        return os;
    }

    void 
    print(std::ostream &os = std::cout) {
        int r = Eigen::Matrix<T,R,C>::rows();
        int c = Eigen::Matrix<T,R,C>::cols();
        os<<"Mat ("<<r<<"x"<<c<<")\n";
        for(int i = 0; i < r; ++i) {
            os << std::endl;
            for(int j = 0; j < c; ++j) {
                os << std::right << std::setw(20) << Eigen::Matrix<T,R,C>::data()[i+j*r];
            }
        }
        os<<std::endl;
    }



    bool   toVar(std::string name, MatIO::var_t &var) {
        Eigen::Matrix<T,R,C> &m = *this;
        return var.set(name, m);
    }
    bool fromVar(MatIO::var_t &var) {
        if(var.empty())
            return false;
        Eigen::Matrix<T,R,C> &m = *this;
        return var.get(m);
    }

};


typedef Mat<double,2,2> Mat2d;
typedef Mat<double,3,3> Mat3d;
typedef Mat<double,4,4> Mat4d;

typedef Mat<int,2,2> Mat2i;
typedef Mat<int,3,3> Mat3i;
typedef Mat<int,4,4> Mat4i;

typedef Mat<float,2,2> Mat2f; 
typedef Mat<float,3,3> Mat3f; 
typedef Mat<float,4,4> Mat4f; 

}}