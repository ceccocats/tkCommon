#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk{ namespace data{

    class ImageData : public ImageData_gen{
        public:

        ImageData(){
            data.useGPU();
        }

        ImageData(const ImageData& s){
            ImageData();
            width = s.width;
            height = s.height;
            channels = s.channels;
            data = s.data;
        }

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
            data.resize(width*height*channels);
            header.name = "image";
            unlockWrite();

        }

        ImageData& operator=(ImageData& s){
 
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            s.lockRead();
            memcpy(data.data(), s.data.data(), width * height * channels);
            s.unlockRead();
            return *this;
        }

        ImageData& operator=(ImageData_gen& s){
 
            ImageData_gen::operator=(s);
            return *this;
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data.size() == 0; }

        void release(){
            if(empty())
                return;

            lockWrite();
            data.resize(0);
            width = 0;
            height = 0;
            channels = 0;
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
                data.fromVar(var["data"]);
            }
            return ok;
        }
        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            ImageData_gen::toVar("info", structVars[0]);
            data.toVar("data", structVars[1]);
            return var.setStruct(name, structVars);
        }

    };
}}