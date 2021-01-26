
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
template<class T, int R = -1, int C = -1>
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
    MatSimple<T, true> gpu;

    Mat(void) {
        init();
    }

    Mat(float *data_h, float *data_d, int r, int c) {
        init();
        if(data_h != nullptr)
            cpu = MatSimple<T, false>(data_h, r, c);
        if(data_d != nullptr) {
            _use_gpu = true;
            gpu = MatSimple<T, true>(data_d, r, c);
        }
    }

    template<typename OtherDerived>
    Mat(const Eigen::MatrixBase<OtherDerived>& other) : Eigen::Matrix<T, R, C>(other) { 
        init();
    }

    template<typename OtherDerived>
    Mat& operator=(const Eigen::MatrixBase <OtherDerived>& other) {
        cpu.resize(other.rows(), other.cols());
        Eigen::Map<Eigen::Matrix<T, R, C>> m(cpu.data, cpu.rows, cpu.cols);
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
        tkASSERT(R == -1 && C == -1, "You cant resize a static Mat");
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

    __host__ Mat<T,R,C>& 
    operator=(const Mat<T,R,C>& s) {
        cpu = s.cpu;
        if(_use_gpu) {
            this->synchGPU();
        }
        return *this;
    }

    __host__
    const Eigen::Map<Eigen::Matrix<T, R, C>> matrix() {
        return Eigen::Map<Eigen::Matrix<T, R, C>>(cpu.data, cpu.rows, cpu.cols);
    }
    
    __host__
    Eigen::Map<Eigen::Matrix<T, R, C>> writableMatrix() {
        return Eigen::Map<Eigen::Matrix<T, R, C>>(cpu.data, cpu.rows, cpu.cols);
    }

    friend std::ostream& 
    operator<<(std::ostream& os, Mat<T,R,C>& s) {
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