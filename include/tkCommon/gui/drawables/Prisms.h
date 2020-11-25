#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/geometry.h"

namespace tk{ namespace gui{

	class Prisms : public Drawable {
        private:
            tk::common::Prisms prisms;
            tk::common::Prisms* ref;
            uint32_t counter = 0;
        public:

            tk::gui::Color_t color;

            Prisms(tk::gui::Color_t color = tk::gui::color::RED){
                update  = false;
                this->color = color;
            }

            Prisms(tk::common::Prisms& prisms,tk::gui::Color_t color = tk::gui::color::RED){
                ref     = &prisms;
                update  = true;

                this->color = color;
            }

            ~Prisms(){

            }

            void updateData(tk::common::Prisms& prisms){
                ref     = &prisms;
                update  = true;
            }

            void onInit(tk::gui::Viewer *viewer){
            }

            void draw(tk::gui::Viewer *viewer){
                if(ref->isChanged(counter) || update){
                    update      = false;

                    ref->lockRead();
                    prisms.data.resize(ref->data.size());
                    for(int i = 0; i < ref->data.size(); i++){
                        prisms.data[i].height = ref->data[i].height;
                        prisms.data[i].points = ref->data[i].points;
                        prisms.data[i].base_z = ref->data[i].base_z;
                    }
                    ref->unlockRead();
                }

                for(int i = 0; i < prisms.data.size(); i++){
                    auto points = prisms.data[i].points;
                    auto base_z = prisms.data[i].base_z;
                    auto height = prisms.data[i].height;

                    glPushMatrix();{
                        glDepthMask(GL_FALSE);
                        glColor4f(color.r(),color.g(),color.b(),color.a());
                        glBegin(GL_POLYGON);
                        for(int i = points.size()-1;i >=0; i--){
                            glVertex3f(points[i].x, points[i].y, base_z);
                        }
                        glEnd();
                        for(int i = 0; i < points.size(); i++){
                            glBegin(GL_POLYGON);
                            glVertex3f(points[i].x, points[i].y, base_z);
                            glVertex3f(points[i].x, points[i].y, base_z+height);
                            glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z+height);
                            glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z);
                            glEnd();
                        }
                        glBegin(GL_POLYGON);
                        for(int i = points.size()-1;i >=0; i--){
                            glVertex3f(points[i].x, points[i].y, base_z + height);
                        }
                        glEnd();
                        glDepthMask(GL_TRUE);

                        glColor4f(color.r(),color.g(),color.b(),color.a());
                        glLineWidth(2);
                        glBegin(GL_LINES);
                        for(int i = 0 ;i <points.size(); i++){
                            glVertex3f(points[i].x, points[i].y, base_z);
                            glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z);
                        }
                        for(int i = 0; i < points.size(); i++){
                            glVertex3f(points[i].x, points[i].y, base_z);
                            glVertex3f(points[i].x, points[i].y, base_z+height);
                        }
                        for(int i = 0 ;i <points.size(); i++){
                            glVertex3f(points[i].x, points[i].y, base_z+height);
                            glVertex3f(points[(i+1)%points.size()].x, points[(i+1)%points.size()].y, base_z+height);
                        }
                        glEnd();
                    }glPopMatrix();
                }
            }

            void imGuiInfos(){
                ImGui::Text("3D prisms drawable");
            }

            void imGuiSettings(){
                ImGui::ColorEdit4("Color", color.color);
            }

            void onClose(){
            }

            std::string toString(){
                return "Prisms";
            }
	};
}}