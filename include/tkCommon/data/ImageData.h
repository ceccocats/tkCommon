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
            mtx->unlock();
        }

        ImageData() {
            mtx = new std::mutex();
        }

        ~ImageData(){
            release();
            delete mtx;
        }


        void onInit(tk::gui::Viewer *viewer){
            gl_texture_shader.init();
            gl_texture.init(width, height, channels);

            float s = 0.5;
            float vertices[] = {
                //positions     //texture cords
                -s, s, 0.0f,   	0.0f, 0.0f, 
                -s,-s, 0.0f,   	0.0f, 1.0f,
                 s,-s, 0.0f,   	1.0f, 1.0f,
                 s, s, 0.0f,   	1.0f, 0.0f
            };
            unsigned int indices[] = {  
                0, 1, 2, // first triangle
                0, 3, 2  // second triangle
            };
            gl_buffer.init();
            gl_buffer.setData(vertices,21);
            gl_buffer.setIndexVector(indices,6);	
        }

        void onAdd(tk::gui::Viewer *viewer){
		    gl_texture.setData(data);
        }

        void draw(tk::gui::Viewer *viewer){
            gl_texture_shader.draw(&gl_texture,&gl_buffer,6); //2 triangles = 6 vertex
        }

        void onClose(){
            gl_texture.release();
            gl_buffer.release();
            gl_texture_shader.close();
        }

    };
}}