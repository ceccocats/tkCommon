#pragma once
#include "tkCommon/gui/drawables/Drawable.h"

namespace tk{ namespace gui{

	class DrawBuffer : public Drawable {

        std::string name;

        public:
            std::vector<tk::gui::Drawable*> drawables;
            
            DrawBuffer(){
                this->name = "DrawBuffer";
            }

            DrawBuffer(std::string name){
                this->name = name;
            }

            DrawBuffer(std::string name, int n, ...){
                va_list arguments; 

                va_start(arguments, n);   
                this->drawables.resize(n);

                for(int i = 0; i < n; i++){
                    this->drawables[i]  = va_arg(arguments, tk::gui::Drawable*);
                }
                va_end(arguments);
                this->name = name;
            }

            ~DrawBuffer(){
                for(int i = 0; i < drawables.size(); i++){
                    delete drawables[i];
                }
            }

            void onInit(tk::gui::Viewer *viewer){
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->onInit(viewer);
                }
            }

            void beforeDraw(tk::gui::Viewer *viewer) {
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->beforeDraw(viewer);
                }            
            }


            void draw(tk::gui::Viewer *viewer){
                for(int i = 0; i < drawables.size(); i++){
                    glPushMatrix();
                    glMultMatrixf(drawables[i]->tf.matrix().data());
                    drawables[i]->draw(viewer);
                    glPopMatrix();
                }
            }

            void imGuiInfos(){
                //for(int i = 0; i < drawables.size(); i++){
                //    drawables[i]->imGuiInfos();
                //}
            }

            void imGuiSettings(){
                //for(int i = 0; i < drawables.size(); i++){
                //    drawables[i]->imGuiSettings();
                //}
            }

            void onClose(){
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->onClose();
                }
            }

            std::string toString(){
                return name;
            }
	};
}}