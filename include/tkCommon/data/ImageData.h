#pragma once
#include <tkCommon/common.h>

namespace tk{namespace data{

    template <class T>
    struct ImageData_t{

        T*  data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;

        void init(int w, int h, int ch){
            width = w;
            height = h;
            channels = ch;
            data = new T[width*height*channels];
        }

        bool empty() {return channels == 0 || width == 0 || height == 0; }

        ImageData_t<T>& operator=(const ImageData_t<T>& s){
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            memcpy(data, s.data, width * height * channels * sizeof(T));

            return *this;
        }

        void release(){
            if(empty())
                return;

            T* tmp = data;
            data = nullptr;
            width = 0;
            height = 0;
            channels = 0;
            delete [] tmp;
        }

        ~ImageData_t(){
            release();
        }
    };
}}