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
            gl_texture.init(width, height, channels);
        }

        void onAdd(tk::gui::Viewer *viewer){
		    gl_texture.setData(data);
        }

        void draw(tk::gui::Viewer *viewer){
            ImGui::Begin("Image", NULL, ImGuiWindowFlags_NoScrollbar);
            int imgX = ImGui::GetWindowSize().x-20;
            int imgY = ImGui::GetWindowSize().y-35;
            ImGui::Image((void*)(intptr_t)gl_texture.texture, ImVec2(imgX, imgY));
            ImGui::End();
        }

        void onClose(){
            gl_texture.release();
            gl_buffer.release();
            gl_texture_shader.close();
        }

    };
}}