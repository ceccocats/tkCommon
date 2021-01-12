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
            header.name = sensorType::CAMDATA;
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

        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            bool ok = ImageData_gen::fromVar(var["info"]);
            tk::data::HeaderData tmp = header; // header will be overwrited by init
            if(ok) {
                init(width, height, channels);
                header = tmp;
                std::vector<uint8_t> values;
                var["data"].get(values);
                memcpy(data, values.data(), values.size()*sizeof(uint8_t));
            }
            return ok;
        }
        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            ImageData_gen::toVar("info", structVars[0]);
            std::vector<uint8_t> values(data, data + height*width*channels);
            structVars[1].set("data", values);
            return var.setStruct(name, structVars);
        }

    };
}}