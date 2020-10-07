#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class colorCloud4f : Drawable {

        private:
            tk::data::CloudData*    cloud;
            tk::gui::Buffer<float>  glbuffer;

        protected:
            void onChange(tk::gui::Viewer *viewer){
                glbuffer.setData(cloud->data_h,cloud->size());
                cloud->modified = false;
            }

        public:
            tk::gui::Color_t        color;

            colorCloud4f(tk::data::CloudData* cloud, tk::gui::Color_t color = tk::gui::color::WHITE){
                this->cloud = cloud;
                this->color = color;
            }

            ~colorCloud4f(){

            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::pointcloud4f();
                glbuffer.init();
            }

            void draw(tk::gui::Viewer *viewer){
                if(cloud->isChange()){
                    onChange(viewer);
                }
                shader->draw(&glbuffer, glbuffer.size(),color);		
            }

            void imGuiSettings(){
                ImGui::Text("colorCloud4f imgui settings");
            }

            void imGuiInfos(){
                ImGui::Text("colorCloud4f imgui infos -> gatty gay");
            }

            void onClose(){
                shader->close();
                glbuffer.release();
                delete shader;
            }
	};
}}