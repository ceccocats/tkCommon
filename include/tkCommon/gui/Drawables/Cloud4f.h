#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloud4f.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4f : public Drawable {

        private:
            tk::data::CloudData*    cloud;
            tk::gui::Buffer<float>  glbuffer;

            float                   imgui_color[4];
            float                   pointSize = 1.0f;

        public:
            tk::gui::Color_t        color;

            Cloud4f(tk::data::CloudData* cloud, tk::gui::Color_t color = tk::gui::color::WHITE){
                this->cloud = cloud;
                this->color = color;           
            }

            ~Cloud4f(){

            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::pointcloud4f();
                glbuffer.init();

                imgui_color[0] = color.r/255.0f;
                imgui_color[1] = color.g/255.0f;
                imgui_color[2] = color.b/255.0f;
                imgui_color[3] = color.a/255.0f; 
            }

            void draw(tk::gui::Viewer *viewer){
                if(cloud->isChanged()){
                    cloud->lock();
                    glbuffer.setData(cloud->points.data_h,cloud->points.size());
                    cloud->unlockRead();
                }

                tk::gui::shader::pointcloud4f* shaderCloud = (tk::gui::shader::pointcloud4f*) shader;
                glPointSize(pointSize);
                shaderCloud->draw(&glbuffer, glbuffer.size()/4,color);
                glPointSize(1.0);		
            }

            void imGuiSettings(){
                ImGui::ColorEdit4("Color", imgui_color);
                ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
                color.r = 255 * imgui_color[0];
                color.g = 255 * imgui_color[1];
                color.b = 255 * imgui_color[2];
                color.a = 255 * imgui_color[3];
            }

            void imGuiInfos(){
                ImGui::Text("Pointcloud has %d points",glbuffer.size());
            }

            void onClose(){
                tk::gui::shader::pointcloud4f* shaderCloud = (tk::gui::shader::pointcloud4f*) shader;
                shaderCloud->close();
                glbuffer.release();
                delete shader;
            }

            std::string toString(){
                return cloud->header.name;
            }
	};
}}