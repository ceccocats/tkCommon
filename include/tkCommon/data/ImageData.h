#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk{ namespace data{

    template<class T>
    class ImageDataX: public tk::data::ImageData_gen<T>{
        public:

        ImageDataX() : ImageData_gen<T>(){
            this->data.useGPU();
        }

        ImageDataX(const ImageDataX<T>& s)  : ImageData_gen<T>() {
            ImageDataX();
            this->width = s.width;
            this->height = s.height;
            this->channels = s.channels;
            this->data = s.data;
        }

        void init(){

            this->lockWrite();
        	tk::data::ImageData_gen<T>::init();
            this->unlockWrite();

        }
        
        void init(int w, int h, int ch){

            this->lockWrite();
        	tk::data::ImageData_gen<T>::init();
            this->width = w;
            this->height = h;
            this->channels = ch;
            this->data.resize(this->width*this->height*this->channels);
            this->header.name = sensorType::CAMDATA;
            this->unlockWrite();

        }

        ImageDataX<T>& operator=(ImageDataX<T>& s){
 
            if(s.width != this->width || s.height != this->height || s.channels != this->channels){
                release();
                init(s.width, s.height, s.channels);
            }
            s.lockRead();
            memcpy(this->data.data(), s.data.data(), this->width * this->height * this->channels * sizeof(T));
            s.unlockRead();
            return *this;
        }

        ImageDataX<T>& operator=(ImageData_gen<T>& s){
 
            ImageData_gen<T>::operator=(s);
            return *this;
        }

        bool empty() {return this->channels == 0 || this->width == 0 || this->height == 0 || this->data.size() == 0; }

        void release(){
            if(empty())
                return;

            this->lockWrite();
            this->data.resize(0);
            this->width = 0;
            this->height = 0;
            this->channels = 0;
            this->unlockWrite();
        }

        uint8_t* at(int row, int col){
            return &this->data[(row*this->width + col) * this->channels];
        }


        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            bool ok = tk::data::ImageData_gen<T>::fromVar(var["info"]);
            tk::data::HeaderData tmp = this->header; // header will be overwrited by init
            if(ok) {
                init(this->width, this->height, this->channels);
                this->header = tmp;
                this->data.fromVar(var["data"]);
            }
            return ok;
        }
        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            tk::data::ImageData_gen<T>::toVar("info", structVars[0]);
            this->data.toVar("data", structVars[1]);
            return var.setStruct(name, structVars);
        }

    };

typedef tk::data::ImageDataX<uint8_t>   ImageData;
typedef tk::data::ImageDataX<float>     ImageDataF;
typedef tk::data::ImageDataX<uint8_t>   ImageDataU8;
typedef tk::data::ImageDataX<uint16_t>  ImageDataU16;
}}