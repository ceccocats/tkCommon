#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

#ifdef ROS_ENABLED
#include <sensor_msgs/Image.h>
#endif

namespace tk{ namespace data{

    template<class T>
    class ImageDataX: public tk::data::ImageData_gen<T>{
        public:

        ImageDataX() : ImageData_gen<T>(){
            this->data.useGPU();

            if(this->T_type.id == tk::data::UINT8){
                this->header.type = tk::data::DataType::IMAGEU8;
            }
            if(this->T_type.id == tk::data::UINT16){
                this->header.type = tk::data::DataType::IMAGEU16;
            }
            if(this->T_type.id == tk::data::FLOAT){
                this->header.type = tk::data::DataType::IMAGEF;
            }
        }

        ImageDataX(const ImageDataX<T>& s)  : ImageData_gen<T>() {
            ImageDataX();
            this->width = s.width;
            this->height = s.height;
            this->channels = s.channels;
            this->data = s.data;
        }

        void init(){

            //this->lockWrite();
        	//tk::data::ImageData_gen<T>::init();
            //this->unlockWrite();

        }
        
        void init(int w, int h, int ch){

            //this->try_lock();
        	//tk::data::ImageData_gen<T>::init();
            this->width = w;
            this->height = h;
            this->channels = ch;
            this->data.resize(this->width*this->height*this->channels);
            //this->unlockWrite();

        }

        ImageDataX<T>& operator=(ImageDataX<T>& s){
 
            if(s.width != this->width || s.height != this->height || s.channels != this->channels){
                release();
                init(s.width, s.height, s.channels);
            }
            //s.lockRead();
            memcpy(this->data.data(), s.data.data(), this->width * this->height * this->channels * sizeof(T));
            //s.unlockRead();
            return *this;
        }

        ImageDataX<T>& operator=(const ImageData_gen<T>& s){
 
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

        T* at(int row, int col){
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


#ifdef ROS_ENABLED
        void toRos(sensor_msgs::Image &msg) {
            tkASSERT(sizeof(T) == sizeof(uint8_t))
            this->header.toRos(msg.header);
            msg.width  = this->width;
            msg.height = this->height;
            if(this->channels == 4)
                msg.encoding = "rgba8";
            if(this->channels == 3)
                msg.encoding = "rgb8";
            if(this->channels == 1)
                msg.encoding = "mono8";

            msg.step = this->width*this->channels*sizeof(T);
            msg.data.resize(this->width * this->height * this->channels);
            memcpy(msg.data.data(), this->data.cpu.data, this->width * this->height * this->channels * sizeof(T));                
        }

        void fromRos(sensor_msgs::Image &msg) {
            tkASSERT(sizeof(T) == sizeof(uint8_t))
            this->header.fromRos(msg.header);
            this->header.type   = DataType::IMAGEU8; 

            int width  = msg.width;
            int height = msg.height;
            int channels = 0;
            channels = msg.encoding == "rgba8"? 4 : channels;
            channels = msg.encoding == "rgb8"? 3 : channels;
            channels = msg.encoding == "mono8"? 1 : channels;
            init(width, height, channels);
            memcpy(this->data, msg.data.data(), width*height*channels*sizeof(uint8_t));
        }
#endif

    };

typedef tk::data::ImageDataX<uint8_t>   ImageData;
typedef tk::data::ImageDataX<float>     ImageDataF;
typedef tk::data::ImageDataX<uint8_t>   ImageDataU8;
typedef tk::data::ImageDataX<uint16_t>  ImageDataU16;
}}