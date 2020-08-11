#pragma once
#include <mutex>
#include <tkCommon/common.h>
#include "SensorData.h"

namespace tk{namespace data{

    template <class T>
    class ImageData : public SensorData{

    public:
        T*  data = nullptr;
        int width = 0;
        int height = 0;
        int channels = 0;
        std::mutex *mtx = nullptr; // mutex contructor is marked delete, so you cant copy the struct containing mutex

        bool gen_tex = false;
        unsigned int texture;
        int index = 0;

        void init(){
            SensorData::init();
            header.name = sensorName::CAMDATA;
        }

        void init(int w, int h, int ch){
        	SensorData::init();

            mtx->lock();
            width = w;
            height = h;
            channels = ch;
            data = new T[width*height*channels];
            mtx->unlock();
            header.name = sensorName::CAMDATA;
        }

        bool empty() {return channels == 0 || width == 0 || height == 0 || data == nullptr; }

		bool checkDimension(SensorData *s){
			auto *t = dynamic_cast<ImageData<T>*>(s);
			if(t->width != width || t->height != height || t->channels != channels){
				return false;
			}
        	return true;
        }

        ImageData<T>& operator=(const ImageData<T>& s){
        	SensorData::operator=(s);
			index = s.index;
            //if(s.width != width || s.height != height || s.channels != channels){
            //    release();
            //    init(s.width, s.height, s.channels);
            //}
            mtx->lock();
            memcpy(data, s.data, width * height * channels * sizeof(T));
            mtx->unlock();

            if(gen_tex == false){
				texture = s.texture;
				gen_tex = s.gen_tex;
            }

            return *this;
        }

        void release(){
            if(empty())
                return;

            mtx->lock();
            T* tmp = data;
            data = nullptr;
            width = 0;
            height = 0;
            channels = 0;
            delete [] tmp;
            glDeleteTextures(1,&texture);
            gen_tex = false;
            mtx->unlock();
        }

        void toGL(){
			if(empty()){
				tk::tformat::printMsg("Viewer","Image empty\n");
			}else{

				//glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
				glBindTexture(GL_TEXTURE_2D, texture);

				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

				// Set texture clamping method
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

				if(this->channels == 4) {
					glTexImage2D(GL_TEXTURE_2D,         // Type of texture
								 0,                   // Pyramid level (for mip-mapping) - 0 is the top level
								 GL_RGB,              // Internal colour format to convert to
								 this->width,          // Image width  i.e. 640 for Kinect in standard mode
								 this->height,          // Image height i.e. 480 for Kinect in standard mode
								 0,                   // Border width in pixels (can either be 1 or 0)
								 GL_RGBA,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
								 GL_UNSIGNED_BYTE,    // Image data type
								 this->data);        // The actual image data itself
				}else if(this->channels == 3){
					glTexImage2D(GL_TEXTURE_2D,         // Type of texture
								 0,                   // Pyramid level (for mip-mapping) - 0 is the top level
								 GL_RGB,              // Internal colour format to convert to
								 this->width,          // Image width  i.e. 640 for Kinect in standard mode
								 this->height,          // Image height i.e. 480 for Kinect in standard mode
								 0,                   // Border width in pixels (can either be 1 or 0)
								 GL_RGB,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
								 GL_UNSIGNED_BYTE,    // Image data type
								 this->data);        // The actual image data itself
				}
			}
        };

        ImageData() {
            mtx = new std::mutex();
        }

        ~ImageData(){
            release();
            delete mtx;
        }

		void draw2D(tk::gui::Viewer *viewer) {

        	if(!gen_tex){
				gen_tex = true;
				glGenTextures(1, &texture);
        	}
			this->toGL();

        	if(tk::gui::Viewer::image_width != this->width)
				tk::gui::Viewer::image_width = this->width;
			if(tk::gui::Viewer::image_height != this->height)
				tk::gui::Viewer::image_height = this->height;

			viewer->tkDrawTextureImage(texture, index);
        }

    };
}}