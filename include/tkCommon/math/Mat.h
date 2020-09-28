
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

namespace tk { namespace math {

/**
 * @brief   Matrix in cuda. Matrix in COLS order.
 * @tparam  T matrix class type
 */
template<class T>
class Mat {

    private:
        static const int MAXSIZE_MARGIN = 0;
        int     _maxSize = 0;
        int     _rows = 0;
        int     _cols = 0;
        int     _size = 0;

    public:
        T*      data_d = nullptr;
        T*      data_h = nullptr;

        Mat(int r=0, int c=0) {
            //std::cout<<"init\n";
            if(r > 0 && c > 0)
                resize(r,c);
        }
        ~Mat() {
            release();
        }

        void copyFrom(T*data, int r, int c) {
            resize(r, c);
            memcpy(data_h, data, _size*sizeof(T));
            synchGPU();
        }

        void copyTo(T*data) {
            memcpy(data, data_h, _size*sizeof(T)); 
        }

        void cloneGPU(const Mat<T> &m) {
            resize(m.rows(), m.cols());
            HANDLE_ERROR( cudaMemcpy(data_d, m.data_d, m.size() * sizeof(T), cudaMemcpyDeviceToDevice) ); 
        }

        void cloneCPU(const Mat<T> &m) {
            resize(m.rows(), m.cols());
            memcpy(data_h, m.data_h, m.size() * sizeof(T)); 
        }
        
        void synchGPU(){ 
            HANDLE_ERROR( cudaMemcpy(data_d, data_h, _size * sizeof(T), cudaMemcpyHostToDevice) ); 
        }

        void synchCPU(){ 
            HANDLE_ERROR( cudaMemcpy(data_h, data_d, _size * sizeof(T), cudaMemcpyDeviceToHost) ); 
        }
        
        void resize(int r, int c) {
            if(data_h == nullptr || data_d == nullptr || _maxSize < r * c){
                _maxSize = r * c + MAXSIZE_MARGIN;
                HANDLE_ERROR( cudaFreeHost(data_h) );
                HANDLE_ERROR( cudaMallocHost(&data_h, _maxSize * sizeof(T)) );
                HANDLE_ERROR( cudaFree(data_d) );
                HANDLE_ERROR( cudaMalloc(&data_d, _maxSize * sizeof(T)) );
            }
            _rows = r;
            _cols = c;
            _size = r * c;
        }

        void release(){
            //std::cout<<"release\n";
            if(data_h != nullptr)
                HANDLE_ERROR( cudaFreeHost(data_h) ); 
            if(data_d != nullptr)
                HANDLE_ERROR( cudaFree(data_d) ); 
        }

        int rows() const { return _rows; }
        int cols() const { return _cols; }
        int size() const { return _size; }

        T& at(int r, int c) { return data_h[r+c*_rows]; }

        void print(std::string name = "") {
            std::cout<<"Mat "<<name<<" ("<<_rows<<"x"<<_cols<<")\n";
            for(int i = 0; i < _rows; ++i) {
                for(int j = 0; j < _cols; ++j) {
                    std::cout << std::right << std::setw(20) << at(i,j);
                }
                std::cout << std::endl;
            }
        }

        /**
         * Filled in row format
         */
        void set(std::initializer_list<T> a_args) {
            if(a_args.size() != size()) {
                std::cout<<"ERROR: you must set all data\n";
                std::cout<<"try insert: "<<a_args.size()<<" in size: "<<size()<<"\n";
                exit(1);
            }
            int i=0;
            for(auto a : a_args) {
                at(i/_cols, i%_cols) = a;
                i++;
            }
            synchGPU();
        }
};

}}