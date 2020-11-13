
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
class MatBase : public tk::math::MatDump {

    protected:
        const float MAXSIZE_MARGIN = 1.5f;
        int     _maxSize;
        int     _maxSizeGPU;
        int     _rows;
        int     _cols;
        int     _size;
        bool    _gpu;

    public:
        T*      data_d;
        T*      data_h;

        __host__
        MatBase(){
            data_d = nullptr;
            data_h = nullptr;
            _maxSize = 0;
            _maxSizeGPU = 0;
            _rows = 0;
            _cols = 0;
            _size = 0;
            _gpu  = false;
        }

        __host__
        ~MatBase(){
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
        MatBase(int r, int c){
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
        cloneGPU(const MatBase<T> &m) {

            tkASSERT(m.hasGPU() == true, "Could not clone from GPU")

            resize(m.rows(), m.cols());
            useGPU();
            HANDLE_ERROR( cudaMemcpy(data_d, m.data_d, m.size() * sizeof(T), cudaMemcpyDeviceToDevice) ); 
        }

        __host__ void 
        cloneCPU(const MatBase<T> &m) {

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

            tkASSERT(_gpu == true, "You set mat only on CPU\n");
            HANDLE_ERROR( cudaMemcpy(data_h, data_d, _size * sizeof(T), cudaMemcpyDeviceToHost) ); 
        }
        
        __host__ void 
        resize(int r, int c) {

            if(_maxSize < r * c){
                _maxSize = (r * c) * MAXSIZE_MARGIN;
                if(data_h != nullptr)
                    delete [] data_h;
                data_h = new T[_maxSize];
            }

            if(_gpu == true){
                if(_maxSizeGPU < r * c) {
                    _maxSizeGPU = (r * c) * MAXSIZE_MARGIN;
                    if(data_d != nullptr)
                        HANDLE_ERROR( cudaFree(data_d) );
                    HANDLE_ERROR( cudaMalloc(&data_d, _maxSizeGPU * sizeof(T)) );
                }        
                tkASSERT(_maxSizeGPU == _maxSize);
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
            //tkASSERT(r < this->_rows && r >= 0, "Out of memory, rows "+std::to_string(r)+" > "+std::to_string(this->_rows)+"\n")
            //tkASSERT(c < this->_cols && c >= 0, "Out of memory, cols "+std::to_string(c)+" > "+std::to_string(this->_cols)+"\n")
            return data_h[r+c*_rows]; 
        }
        
        __device__ T&  
        atGPU(int r, int c) { 
            //tkASSERT(_gpu == true, "You set mat only on CPU\n");
            //tkASSERT(r < this->_rows && r >= 0, "Out of memory, rows "+std::to_string(r)+" > "+std::to_string(this->_rows)+"\n")
            //tkASSERT(c < this->_cols && c >= 0, "Out of memory, cols "+std::to_string(c)+" > "+std::to_string(this->_cols)+"\n")    
            return data_d[r+c*_rows];
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

        __host__ void
        fill(T value) {
            for (int i = 0; i < _size; i++) {
                data_h[i] = value;
            }

            if(_gpu == true){
                synchGPU();
            }
        }

        __host__ MatBase<T>& 
        operator=(const MatBase<T>& s) {
            cloneCPU(s);
            if(s.hasGPU() == true)
                cloneGPU(s);
            return *this;
        }

        friend std::ostream& 
        operator<<(std::ostream& os, const MatBase& s) {
            os<<"Mat ("<<s.rows()<<"x"<<s.cols()<<")";
            return os;
        }

        void 
        print() {
            for(int i = 0; i < rows(); ++i) {
                std::cout << std::endl;
                for(int j = 0; j < cols(); ++j) {
                    std::cout << std::right << std::setw(20) << data_h[i+j*rows()];
                }
            }
            std::cout<<std::endl;
        }

};


// spicialization for Mat<something>
template<typename T, typename Enable = void>
class Mat : public MatBase<T> { 
public:
};
// spicialization for Mat<float> and other arithme
template<typename T>
class Mat<T, typename std::enable_if<std::is_arithmetic<T>::value>::type> : public MatBase<T> { 
public:
    bool toVar(std::string name, MatIO::var_t &var) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
        mat.resize(this->rows(), this->cols());
        memcpy(mat.data(), this->data_h, sizeof(T)*this->rows()*this->cols());
        return var.set(name, mat);
    }
    bool fromVar(MatIO::var_t &var) {
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat;
        if(!var.get(mat))
            return false;
        this->resize(mat.rows(), mat.cols());
        memcpy(this->data_h, mat.data(), sizeof(T)*this->rows()*this->cols());
        return true;
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