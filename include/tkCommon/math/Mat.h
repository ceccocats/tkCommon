
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
#include "tkCommon/math/MatSimple.h"
#include "tkCommon/math/MatIO.h"

namespace tk { namespace math {

// GPU INFO
static int  Mat_device = -1;
static bool Mat_integrated_memory = false;

/**
 * @brief   Matrix in cuda. Matrix in COLS order.
 * @tparam  T matrix class type
 */
template<class T>
class Mat : public tk::math::MatDump {

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
    MatSimple<T, false> cpu;
    MatSimple<T, true>  gpu;

    Mat(void) {
        init();
    }

    Mat(T *data_h, T *data_d, int r, int c) {
        init();
        if(data_h != nullptr)
            cpu = MatSimple<T, false>(data_h, r, c);
        if(data_d != nullptr) {
            _use_gpu = true;
            gpu = MatSimple<T, true>(data_d, r, c);
        }
    }

    //template<typename OtherDerived>
    //Mat(const Eigen::MatrixBase<OtherDerived>& other) : Eigen::Matrix<T,-1,-1>(other) { 
    //    init();
    //}

    template<typename OtherDerived>
    Mat<T>& operator=(const Eigen::MatrixBase <OtherDerived>& other) {
        cpu.resize(other.rows(), other.cols());
        Eigen::Map<Eigen::Matrix<T,-1,-1>> m(cpu.data, cpu.rows, cpu.cols);
        m = other;
        return *this;
    }

    __host__
    ~Mat() {
        cpu.release();
        gpu.release();
    }


    __host__ void
    useGPU(){
        _use_gpu = true;
        resize(rows(),cols());
    }

    __host__ bool
    hasGPU() const {
        return _use_gpu;
    }

    __host__ void 
    copyFrom(T*data, int r, int c) {
        resize(r, c);
        memcpy(cpu.data, data, size()*sizeof(T));
        if(_use_gpu == true){
            synchGPU();
        }
    }

    __host__ void 
    copyTo(T*data) {
        memcpy(data, cpu.data, size()*sizeof(T)); 
    }
    
    __host__ void 
    synchGPU(){ 
        useGPU();
        tkCUDA( cudaMemcpy(gpu.data, cpu.data, size() * sizeof(T), cudaMemcpyHostToDevice) ); 
    }

    __host__ void 
    synchCPU(){ 
        tkASSERT(_use_gpu == true, "You set mat only on CPU\n");
        resize(gpu.rows, gpu.cols);
        tkCUDA( cudaMemcpy(cpu.data, gpu.data, size() * sizeof(T), cudaMemcpyDeviceToHost) ); 
    }
    
    __host__ int
    rows() { return cpu.rows; }

    __host__ int
    cols() { return cpu.cols; }

    __host__ int
    size() { return cpu.size; }

    __host__ T*
    data() { return cpu.data; }

    __host__ void 
    resize(int r, int c) {
        cpu.resize(r,c);
        if(_use_gpu == true){
            gpu.resize(r, c);
        }
    }


    __host__ T&
    operator()(int i, int j) {
        return cpu.data[i+j*cpu.rows]; 
    }

    __host__ T&
    operator[](int n) {
        return cpu.data[n]; 
    }

    __host__ Mat<T>& 
    operator=(const Mat<T>& s) {
        cpu = s.cpu;
        if(_use_gpu) {
            this->synchGPU();
        }
        return *this;
    }

    __host__
    const Eigen::Map<Eigen::Matrix<T,-1,-1>> matrix() {
        return Eigen::Map<Eigen::Matrix<T,-1,-1>>(cpu.data, cpu.rows, cpu.cols);
    }
    
    __host__
    Eigen::Map<Eigen::Matrix<T,-1,-1>> writableMatrix() {
        return Eigen::Map<Eigen::Matrix<T,-1,-1>>(cpu.data, cpu.rows, cpu.cols);
    }

    friend std::ostream& 
    operator<<(std::ostream& os, Mat<T>& s) {
        if(s.size() <= 16)
            s.print(os);
        else
            os<<"Mat ("<<s.rows()<<"x"<<s.cols()<<")";
        return os;
    }

    __host__ void 
    print(std::ostream &os = std::cout) {
        int r = rows();
        int c = cols();
        os<<"Mat ("<<r<<"x"<<c<<")\n";
        for(int i = 0; i < r; ++i) {
            os << std::endl;
            for(int j = 0; j < c; ++j) {
                os << std::right << std::setw(20) << cpu.data[i+j*r];
            }
        }
        os<<std::endl;
    }

    __host__
    void fill(float val) {
        for(int i=0; i<cpu.size; i++)
            cpu.data[i] = val;
    }


    bool   toVar(std::string name, MatIO::var_t &var) {
        return var.set(name, cpu);
    }
    bool fromVar(MatIO::var_t &var) {
        if(var.empty())
            return false;
        return var.get(cpu);
    }

};

template<class T, int R, int C>
class MatStatic : public Mat<T> {
    
private:
    T static_data[R*C];

public:
    MatStatic() : Mat<T>(static_data, nullptr, R, C) {
    }

    template<typename OtherDerived>
    Mat<T>& operator=(const Eigen::MatrixBase <OtherDerived>& other) {
        tkASSERT(R == other.rows() && C == other.cols());
        Eigen::Map<Eigen::Matrix<T,-1,-1>> m(this->cpu.data, this->cpu.rows, this->cpu.cols);
        m = other;
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