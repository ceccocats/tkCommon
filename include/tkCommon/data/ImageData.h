#pragma once
#include <mutex>
#include <tkCommon/common.h>

namespace tk{namespace data{

    template <class T>
    struct ImageData_t{

        T*  data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        std::mutex *mtx = nullptr; // mutex contructor is marked delete, so you cant copy the struct containing mutex

        void init(int w, int h, int ch){
            mtx->lock();
            width = w;
            height = h;
            channels = ch;
            data = new T[width*height*channels];
            mtx->unlock();
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

        ImageData_t<T>& operator=(const ImageData_t<T>& s){
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            mtx->lock();
            memcpy(data, s.data, width * height * channels * sizeof(T));
            mtx->unlock();
            return *this;
        }

        void release(){
            if(empty())
                return;

            mtx->lock();
            T* tmp = data;
            data = nullptr;
            width = 0;
            height = 0;
            channels = 0;
            delete [] tmp;
            mtx->unlock();
        }


        ImageData_t() {
            mtx = new std::mutex();
        }

        ~ImageData_t(){
            release();
            delete mtx;
        }

        matvar_t *matVar(std::string name = "image") {
            tkASSERT(sizeof(T) == sizeof(uint8_t))
            size_t dim[3] = { height, width, channels }; // create 1x1 struct
            matvar_t *var = Mat_VarCreate(name.c_str(), MAT_C_UINT8, MAT_T_UINT8, 3, dim, data, 0);
            return var;
        }
    };
}}