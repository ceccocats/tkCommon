#pragma once

#include "tkCommon/data/gen/DepthData_gen.h"

namespace tk{ namespace data{

    class DepthData : public DepthData_gen{
        public:

        DepthData(){
            data.useGPU();
        }

        DepthData(const DepthData& s){
            DepthData();
            width = s.width;
            height = s.height;
            data = s.data;
        }

        void init(){

            lockWrite();
        	DepthData_gen::init();
            unlockWrite();

        }
        
        void init(int w, int h){

            lockWrite();
        	DepthData_gen::init();
            width = w;
            height = h;
            data.resize(width*height);
            header.name = sensorType::CAMDATA;
            unlockWrite();

        }

        DepthData& operator=(DepthData& s){
 
            if(s.width != width || s.height != height ){
                release();
                init(s.width, s.height);
            }
            s.lockRead();
            memcpy(data.cpu.data, s.data.cpu.data, width * height * sizeof(uint16_t));
            s.unlockRead();
            return *this;
        }

        DepthData& operator=(DepthData_gen& s){
 
            DepthData_gen::operator=(s);
            return *this;
        }

        bool empty() {return width == 0 || height == 0 || data.size() == 0; }

        void release(){
            if(empty())
                return;

            lockWrite();
            data.resize(0);
            width = 0;
            height = 0;
            unlockWrite();
        }

        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            bool ok = DepthData_gen::fromVar(var["info"]);
            tk::data::HeaderData tmp = header; // header will be overwrited by init
            if(ok) {
                init(width, height);
                header = tmp;
                data.fromVar(var["data"]);
            }
            return ok;
        }
        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            DepthData_gen::toVar("info", structVars[0]);
            data.toVar("data", structVars[1]);
            return var.setStruct(name, structVars);
        }

    };
}}