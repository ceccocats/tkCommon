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

        matvar_t *toMatVar(std::string name = "image") {
            tkASSERT(sizeof(T) == sizeof(uint8_t))
            tkASSERT(channels == 4)

            size_t dim[3] = { height, width, 3 }; // create 1x1 struct
            matvar_t *var = Mat_VarCreate(name.c_str(), MAT_C_UINT8, MAT_T_UINT8, 3, dim, data, 0);

            // allocated by libmatio
            uint8_t *tmp = (uint8_t *)var->data;

            // RGBA,RGBA,RGBA,... -> RRR...,GGG...,BBB...,
            for(int i=0; i<height; i++)
            for(int j=0; j<width; j++) {
                    tmp[j*height + i + height*width*0] = data[i*width*4 + j*4 + 0];
                    tmp[j*height + i + height*width*1] = data[i*width*4 + j*4 + 1];
                    tmp[j*height + i + height*width*2] = data[i*width*4 + j*4 + 2];
            }
            return var;
        }

		void draw2D(tk::gui::Viewer *viewer) {

			int width = viewer->width;
			int height = viewer->height;
			float xLim = viewer->xLim;
			float yLim = viewer->yLim;

        	if(!gen_tex){
				gen_tex = true;
				glGenTextures(1, &texture);
        	}
			this->toGL();

        	if(tk::gui::Viewer::image_width != this->width)
				tk::gui::Viewer::image_width = this->width;
			if(tk::gui::Viewer::image_height != this->height)
				tk::gui::Viewer::image_height = this->height;

        	float w,h;

			glPushMatrix(); {
				tk::gui::Viewer::tkViewportImage(width, height, xLim, yLim, index, w, h);

				glColor4f(1,1,1,1);
				tk::gui::Viewer::tkDrawTexture(texture, 1, 1);

			} glPopMatrix();

        }
    };
}}