#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/data/GpsData.h"
#include "tkCommon/gui/shader/circle.h"

namespace tk{ namespace gui{

	class Gps : public Drawable {

        private:
            tk::data::GpsData* gps;

            int nPos;
            int lastPos;
            tk::math::VecStatic<tk::gui::shader::circle*,40> circles;

            tk::common::GeodeticConverter geoConv;

            float lineSize = 2.0f;
            bool  update = false;

            bool initted = false;

            std::string name = "";
            std::stringstream print;

        public:
            tk::gui::Color_t        color;

            Gps(int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED){
                this->color = color;  
                this->nPos = nPos;
                this->lastPos = -1;
                this->initted = false;
            }

            Gps(tk::data::GpsData* gps, int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED){
                this->gps = gps;
                this->color = color;  
                this->nPos = nPos;
                this->lastPos = -1;
                this->initted = true;
            }

            ~Gps(){

            }

            void updateRef(tk::data::GpsData* gps){
                this->gps = gps;   
                initted = update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                for(int i = 0; i < circles.size(); i++){
                    circles[i] = new tk::gui::shader::circle();
                }
            }

            void draw(tk::gui::Viewer *viewer){
                if(initted == true){
                    if(gps->isChanged() || update){
                        update = false;

                        gps->lockRead();
                        if(!geoConv.isInitialised() && gps->sats > 10) {
                            geoConv.initialiseReference(gps->lat,gps->lon,gps->heigth);
                        }
                        name = gps->header.name;
                        print.str("");
                        print<<(*gps);
                        double x, y, z;
                        if (geoConv.isInitialised())
                            geoConv.geodetic2Enu(gps->lat,gps->lon,gps->heigth,&x, &y, &z);
                        gps->unlockRead();
                        
                        float RAGGIO = 2.0f; //TODO: BOSI
                        lastPos = (lastPos+1) % nPos;
                        circles[lastPos]->makeCircle(x,y,z,RAGGIO);                
                    }

                    for(int i = 0; i < nPos; i++){
                        circles[i]->draw(color,lineSize);
                    }   	
                }
            }

            void imGuiSettings(){
                ImGui::ColorEdit4("Color", color.color);
                ImGui::SliderFloat("Size",&lineSize,1.0f,20.0f,"%.1f");
                ImGui::SliderInt("Last poses",&nPos,1,40);
            }

            void imGuiInfos(){
                ImGui::Text("%s",print.str().c_str());
            }

            void onClose(){
                for(int i = 0; i < circles.size(); i++){
                    circles[i]->close();
                    delete circles[i];
                }
            }

            std::string toString(){
                return name;
            }
	};
}}