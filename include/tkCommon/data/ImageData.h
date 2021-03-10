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


        }
        
        void init(int w, int h, int ch){

            width = w;
            height = h;
            channels = ch;
            data.resize(width*height*channels);
            header.name = "image";

        }

        ImageData& operator=(const ImageData& s) {
 
            if(s.width != width || s.height != height || s.channels != channels){
                release();
                init(s.width, s.height, s.channels);
            }
            //s.lockRead();
            data = s.data;
            //s.unlockRead();
            return *this;
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data.size() == 0; }

        void release(){
            if(empty())
                return;

            data.resize(0);
            width = 0;
            height = 0;
            channels = 0;
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