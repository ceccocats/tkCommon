#pragma once

#include "tkCommon/data/ImageData.h"

namespace tk{ namespace data{

    class StereoData : public SensorData{
        public:

        // Image with consecutive left and right images
        ImageData data;

        int width, height, channels;
        int c_width, c_height, c_channels;
        int d_width, d_height;

        ImageData left;
        ImageData right;
        ImageData color;
        ImageDataU16 depth;

        void init(int w, int h, int c){
            data.init(w,h*2,c);
            data.data.useGPU();

            width = w;
            height = h;
            channels = c;

            left.data.cpu.data = data.data.cpu.data;
            left.data.gpu.data = data.data.gpu.data;
            right.data.cpu.data = data.data.cpu.data + w*h*c;
            right.data.gpu.data = data.data.gpu.data + w*h*c;

            left.width = right.width = w;
            left.height = right.height = h;
            left.channels = right.channels = c;

        }

        void initColor(int w, int h, int c){
            color.init(w, h, c);
            c_width = w;
            c_height = h;
            c_channels = c;
        }

        void initDepth(int w, int h){
            depth.init(w, h, 1);
            d_width = w;
            d_height = h;

        }

        StereoData& operator=(StereoData& s){
 
            if(s.width != width || s.height != height || s.channels != channels){
                init(s.width, s.height, s.channels);
            }
            s.lockRead();
            data = s.data;
            s.unlockRead();
            return *this;
        }
    };

}}