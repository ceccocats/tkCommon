#pragma once
#include <mutex>
#include <tkCommon/common.h>
#include "SensorData.h"

namespace tk{namespace data{

    template <class T>
    class ImageData : public SensorData{

    public:
        T*  data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        std::mutex *mtx = nullptr; // mutex contructor is marked delete, so you cant copy the struct containing mutex

        void init(){}

        void init(int w, int h, int ch){
        	SensorData::init();

            mtx->lock();
            width = w;
            height = h;
            channels = ch;
            data = new T[width*height*channels];
            mtx->unlock();
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

		bool checkDimension(SensorData *s){
			auto *t = dynamic_cast<ImageData<T>*>(s);
			if(t->width != width || t->height != height || t->channels != channels){
				return false;
			}
        	return true;
        }

        ImageData<T>& operator=(const ImageData<T>& s){
        	SensorData::operator=(s);

            //if(s.width != width || s.height != height || s.channels != channels){
            //    release();
            //    init(s.width, s.height, s.channels);
            //}
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


        ImageData() {
            mtx = new std::mutex();
        }

        ~ImageData(){
            release();
            delete mtx;
        }

        matvar_t *toMatVar(std::string name = "image") {
            tkASSERT(sizeof(T) == sizeof(uint8_t))
            tkASSERT(channels == 4)

            size_t dim[3] = { height, width, 3 }; // create 1x1 struct
            matvar_t *var = Mat_VarCreate(name.c_str(), MAT_C_UINT8, MAT_T_UINT8, 3, dim, data, 0);

            // allocated by libmatio
            uint8_t *tmp = (uint8_t *)var->data;

            // RGBA,RGBA,RGBA,... -> RRR...,GGG...,BBB...,
            for(int i=0; i<height; i++)
            for(int j=0; j<width; j++) {
                    tmp[j*height + i + height*width*0] = data[i*width*4 + j*4 + 0];
                    tmp[j*height + i + height*width*1] = data[i*width*4 + j*4 + 1];
                    tmp[j*height + i + height*width*2] = data[i*width*4 + j*4 + 2];
            }
            return var;
        }

        bool fromMatVar(matvar_t *var) {
            tkASSERT(var->data_type == MAT_T_UINT8);
            tkASSERT(var->rank == 3);
            int h = var->dims[0];
            int w = var->dims[1];
            int c = var->dims[2];
            tkASSERT(h>0);
            tkASSERT(w>0);
            tkASSERT(c==3);
            release();
            init(w, h, 4);
            
            // allocated by libmatio
            uint8_t *tmp = (uint8_t *)var->data;

            // RRR...,GGG...,BBB..., -> RGBA,RGBA,RGBA,...
            #pragma omp parallel for
            for(int i=0; i<height; i++)
            for(int j=0; j<width; j++) {
                    data[i*width*4 + j*4 + 0] = tmp[j*height + i + height*width*0];
                    data[i*width*4 + j*4 + 1] = tmp[j*height + i + height*width*1];
                    data[i*width*4 + j*4 + 2] = tmp[j*height + i + height*width*2];
                    data[i*width*4 + j*4 + 3] = 255;
            }
            return true;
        }
    };
}}