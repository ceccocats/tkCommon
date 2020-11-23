#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk{ namespace data{

    class ImageData : public ImageData_gen{
        public:

        void init(){

            lockWrite();
        	ImageData_gen::init();
            unlockWrite();

        }
        
        void init(int w, int h, int ch){

            lockWrite();
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

            lockWrite();
            s.lockRead();
            memcpy(data, s.data, width * height * channels);
            s.unlockRead();
            unlockWrite();
            return *this;
        }

        ImageData& operator=(ImageData_gen& s){
 
            ImageData_gen::operator=(s);
            return *this;
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

        void release(){
            if(empty())
                return;

            lockWrite();
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