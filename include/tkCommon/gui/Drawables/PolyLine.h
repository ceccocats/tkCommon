#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include <eigen3/Eigen/Dense>

namespace tk{ namespace gui{

    class line : public tk::rt::Lockable{
        public:
            std::vector<Eigen::Vector3f> points;
    };


	class Polyline : public Drawable {

        private:
            tk::gui::line* line;
            tk::gui::Buffer<float> glData;

            float lineSize = 1.0f;

            std::string name;

        public:
            tk::gui::Color_t        color;

            Polyline(tk::gui::line* line, std::string name = "PolyLines", tk::gui::Color_t col = tk::gui::color::RED){
                update      = true;
                this->line  = line;
                this->name = name;
                this->color = col;
                glData.init();
            }

            ~Polyline(){

            }

            void updateRef(tk::gui::line* line){
                this->line = line;   
                update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::linesMonocolor();
            }

            void draw(tk::gui::Viewer *viewer){
                if(line->isChanged() || update){
                    update = false;

                    line->lock();
                    glData.setData((float*)line->points.data(),line->points.size()*3);
                    line->unlockRead();               
                }

                auto shaderLine = (tk::gui::shader::linesMonocolor*)shader;
                shaderLine->draw(&glData,glData.size()/3,lineSize,color,GL_LINE_STRIP);	
            }

            void imGuiSettings(){
                ImGui::ColorEdit4("Color", color.color);
                ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
            }

            void imGuiInfos(){
                ImGui::Text("%s","TODO: in futuro stampare numero punti");
            }

            void onClose(){
                auto shaderLine = (tk::gui::shader::linesMonocolor*)shader;
                shaderLine->close();
                delete shaderLine;
                glData.release();
            }

            std::string toString(){
                return name;
            }
	};
}}