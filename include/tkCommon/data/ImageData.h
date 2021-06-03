#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"
#include "tkCommon/gui/utils/CommonViewer.h"

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

            this->width = 0;
            this->height = 0;
            this->channels = 0;
        }

        ImageDataX(const ImageDataX<T>& s)  : ImageData_gen<T>() {
            ImageDataX();
            this->width = s.width;
            this->height = s.height;
            this->channels = s.channels;
            this->data = s.data;
        }

        void init(){
            
        }
        
        void init(int w, int h, int ch){

            this->width = w;
            this->height = h;
            this->channels = ch;
            this->data.resize(this->width*this->height*this->channels);
        }

        ImageDataX<T>& operator=(ImageDataX<T>& s){

            SensorData::operator=(s);
            if(s.width != this->width || s.height != this->height || s.channels != this->channels){
                init(s.width, s.height, s.channels);
            }
            memcpy(this->data.data(), s.data.data(), this->width * this->height * this->channels * sizeof(T));
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

            this->data.resize(0);
            this->width = 0;
            this->height = 0;
            this->channels = 0;
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

        void toKitty(const std::string& fileName){
            tk::gui::common::writeImagePng(fileName,this->data.cpu.data,this->width,this->height,this->channels);
        }

        void fromKitty(const std::string& fileName){
            int height, width, channels;
            uint8_t* image = tk::gui::common::loadImage(fileName, &width, &height, &channels);
            init(width,height,channels);
            memcpy(this->data.cpu.data, image, height*width*channels*sizeof(uint8_t));
            free(image);
        }


#ifdef ROS_ENABLED
        void toRos(sensor_msgs::Image &msg) {
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
            this->header.fromRos(msg.header);
            int width  = msg.width;
            int height = msg.height;
            
            int channels = 0;
            channels = msg.encoding == "rgba8"? 4 : channels;
            channels = msg.encoding == "rgb8"? 3 : channels;
            channels = msg.encoding == "mono8"? 1 : channels;
            channels = msg.encoding == "32FC1"? 1 : channels;
            tkASSERT(channels != 0, "image encoding not supported")
            init(width, height, channels);
            memcpy(this->data.cpu.data, msg.data.data(), width*height*channels*sizeof(T));
        }
#endif

    };

typedef tk::data::ImageDataX<uint8_t>   ImageData;
typedef tk::data::ImageDataX<float>     ImageDataF;
typedef tk::data::ImageDataX<uint8_t>   ImageDataU8;
typedef tk::data::ImageDataX<uint16_t>  ImageDataU16;
}}