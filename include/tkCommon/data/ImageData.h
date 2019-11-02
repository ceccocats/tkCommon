#pragma once
#include <mutex>
#include <tkCommon/common.h>

#ifdef VIBRANTE

#include <dw/image/Image.h>
#endif

namespace tk{namespace data{

    template <class T>
    struct ImageData_t{

        T*  data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        std::mutex mtx;

#ifdef VIBRANTE
        dwImageHandle_t dwImage;
        dwContextHandle_t *dw = nullptr;
        dwImageProperties props;
#endif

        void init(int w, int h, int ch){
            mtx.lock();
            width = w;
            height = h;
            channels = ch;
            data = new T[width*height*channels];
            mtx.unlock();
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

        ImageData_t<T>& operator=(const ImageData_t<T>& s){
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            mtx.lock();
#ifdef VIBRANTE
            if(dw != s.dw) {
                if(dw != nullptr){
                    dwImage_destroy(dwImage);
                }

                dw = s.dw;
                props = s.props;
                dwImage_create(&dwImage, props, *dw);
                dwImage_copyConvert(dwImage, s.dwImage, *dw);
            }
            else{
                dwImage_copyConvert(dwImage, s.dwImage, *dw);
            }
#endif
            memcpy(data, s.data, width * height * channels * sizeof(T));
            mtx.unlock();
            return *this;
        }

        void copyData(const ImageData_t<T>& s){
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            mtx.lock();
            memcpy(data, s.data, width * height * channels * sizeof(T));
            mtx.unlock();
        }

        void release(){
            if(empty())
                return;

            mtx.lock();
            T* tmp = data;
            data = nullptr;
            width = 0;
            height = 0;
            channels = 0;
            delete [] tmp;
            mtx.unlock();
        }

        ~ImageData_t(){
            release();
        }
    };
}}