#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk{ namespace data{

    class ImageData : public ImageData_gen{
        public:

        void init(){

            lock();
        	ImageData_gen::init();
            unlockWrite();

        }
        
        void init(int w, int h, int ch){

            lock();
        	ImageData_gen::init();
            width = w;
            height = h;
            channels = ch;
            data = new uint8_t[width*height*channels];
            header.name = sensorName::CAMDATA;
            unlockWrite();

        }

        ImageData& operator=(ImageData& s){
 
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }

            lock();
            s.lock();
            memcpy(data, s.data, width * height * channels);
            s.unlockRead();
            unlockWrite();
            return *this;
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

        void release(){
            if(empty())
                return;

            lock();
            uint8_t* tmp = data;
            data = nullptr;
            width = 0;
            height = 0;
            channels = 0;
            delete [] tmp;
            unlockWrite();
        }



    };
}}