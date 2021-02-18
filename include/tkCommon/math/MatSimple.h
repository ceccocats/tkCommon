
#pragma once
#include <iostream>
#include <cstdio>
#include <vector>
#include <time.h>
#include <iomanip>
#include <string>
#include <sstream>
#include <initializer_list>
#include "tkCommon/common.h"
#include "tkCommon/cuda.h"

namespace tk { namespace math {

template<class T, bool CUDA = true>
struct MatSimple {
    T* data;
    int rows, cols;
    int size;
    int maxSize;

    bool owned;             // if true this pointer doesnt belong to the class
    bool integrated_memory; // if true use nvidia integrated memory

    __host__
    MatSimple() {
        data = nullptr;
        rows = cols = size = maxSize = 0;
        integrated_memory = false;
        owned = false;
    }
    
    __host__
    MatSimple(T*data, int r, int c) {
        this->data = data;
        rows = r;
        cols = c;
        size = maxSize = rows*cols;
        integrated_memory = false;
        owned = true;
    }

    __host__ __device__ T&  
    at(int r, int c) { 
        return data[r+c*rows];
    }

    __host__
    void resize(int r, int c) {
        if(r != rows || c != cols) {
            tkASSERT(!owned, "you cant resize an owned Matrix\n");
            if(maxSize < r * c) {
                maxSize = (r * c);

                if(CUDA) {   
                    if(data != nullptr)
                        if(!integrated_memory)
                            tkCUDA( cudaFree(data) );
                        else
                            tkCUDA( cudaFreeHost(data) );

                    if(!integrated_memory)
                        tkCUDA( cudaMalloc(&data, maxSize * sizeof(T)) );
                    else
                        tkCUDA( cudaMallocHost(&data, maxSize * sizeof(T)) );
                } else {
                    if(data != nullptr)
                        delete data;
                    data = new T[maxSize];
                }
            }
            rows = r;
            cols = c;   
            size = r*c;     
        }
    }

    __host__
    void release() {
        if(!owned && data != nullptr) {
            if(CUDA)
                tkCUDA( cudaFree(data) );
            else
                delete data;
        }
    }
    

};

}}