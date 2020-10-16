#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/data/GPSData.h"

namespace tk{ namespace gui{

	class Gps : public Drawable {

        private:
            tk::data::GPSData* gps;

            static const int        lastPos = 40;
            int                     nPos;
            tk::gui::Buffer<float>  glbuffer[lastPos];

            static const int circlePoints = 300;
            tk::math::Mat<tk::gui::shader::linesMonocolor::line_color_t> points;
            tk::math::Mat<tk::gui::shader::linesMonocolor::line_color_t> lines;

            tk::common::GeodeticConverter geoConv;

            float   imgui_color[4];
            float   lineSize = 2.0f;
            float   radius;

            bool update = false;

        public:
            tk::gui::Color_t        color;

            Gps(tk::data::GPSData* gps, int nPos = 10, float radius = 3.0f, tk::gui::Color_t color = tk::gui::color::RED){
                this->gps = gps;
                this->color = color;  
                this->radius = radius;
                this->nPos = nPos;
            }

            ~Gps(){

            }

            void updateRef(tk::data::GPSData* gps){
                this->gps = gps;   
                update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::linesMonocolor();

                imgui_color[0] = color.r/255.0f;
                imgui_color[1] = color.g/255.0f;
                imgui_color[2] = color.b/255.0f;
                imgui_color[3] = color.a/255.0f; 

                lines.resize(1,circlePoints);
                points.resize(1,lastPos);

                for(int i = 0; i < lastPos; i++){
                    glbuffer[i].init();
                    points.atCPU(0,i).x = 0;
                    points.atCPU(0,i).y = 0;
                    points.atCPU(0,i).z = 0;
                }
            }

            void draw(tk::gui::Viewer *viewer){
                if(gps->isChanged() || update){
                    update = false;

                    gps->lock();

                    if(!geoConv.isInitialised()) {
                        geoConv.initialiseReference(gps->lat,gps->lon,gps->heigth);
                    }

                    double x, y, z;
                    geoConv.geodetic2Enu(gps->lat,gps->lon,gps->heigth,&x, &y, &z);

                    // Shifting
                    for(int i = lastPos-1; i > 0; i--){
                        points.atCPU(0,i) = points.atCPU(0,i-1);
                    }
                    points.atCPU(0,0).x = x;
                    points.atCPU(0,0).y = y;
                    points.atCPU(0,0).z = z;


                    for(int j = 0; j < nPos; j++){
                        for(int i = 0; i < circlePoints; i++){
                            float angle = 2.0f * M_PI * (float)i/(float)circlePoints;
                            lines.atCPU(0,i).x = points.atCPU(0,j).x + cos(angle) * radius;
                            lines.atCPU(0,i).y = points.atCPU(0,j).y + sin(angle) * radius;
                            lines.atCPU(0,i).z = points.atCPU(0,j).z;
                        }
                        glbuffer[j].setData((float*)lines.data_h,circlePoints*3);
                    }
                    gps->unlockRead();
                }

                tk::gui::shader::linesMonocolor* shaderLines = (tk::gui::shader::linesMonocolor*) shader;

                tk::gui::Color_t col;
                col.r = color.r;
                col.g = color.g;
                col.b = color.b;
                for(int i = 0; i < nPos; i++){
                    col.a = color.a / ((i+1)/4.0f);
                    shaderLines->draw(&glbuffer[i], glbuffer[i].size()/3, lineSize, col, GL_LINE_LOOP);
                }
                    	
            }

            void imGuiSettings(){
                ImGui::ColorEdit4("Color", imgui_color);
                ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
                if(ImGui::SliderInt("Last n pos",&nPos,1,lastPos)){
                    update = true;
                }
                ImGui::SliderFloat("Radius",&radius,1.0f,40.0f,"%.1f");
                color.r = 255 * imgui_color[0];
                color.g = 255 * imgui_color[1];
                color.b = 255 * imgui_color[2];
                color.a = 255 * imgui_color[3];
            }

            void imGuiInfos(){
                std::stringstream print;
                print<<(*gps);
                ImGui::Text("%s",print.str().c_str());
                print.clear();
            }

            void onClose(){
                tk::gui::shader::linesMonocolor* shaderLines = (tk::gui::shader::linesMonocolor*) shader;
                shaderLines->close();
                for(int i = 0; i < lastPos; i++){
                    glbuffer[i].release();
                }
                delete shader;
            }

            std::string toString(){
                return gps->header.name;
            }
	};
}}