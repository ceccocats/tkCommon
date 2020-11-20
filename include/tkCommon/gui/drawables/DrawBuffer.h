#pragma once
#include "tkCommon/gui/drawables/Drawable.h"

namespace tk{ namespace gui{

	class DrawBuffer : public Drawable {

        std::vector<tk::gui::Drawable*> drawables;
        std::string name;

        public:
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

            void pushDrawable(tk::gui::Drawable* drawable){
                this->drawables.push_back(drawable);
            }

            void updateDrawable(int n, tk::gui::Drawable* drawable){
                tkASSERT(n < drawables.size());
                this->drawables[n] = drawable;
            }

            void draw(tk::gui::Viewer *viewer){
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->draw(viewer);
                }
            }

            void imGuiInfos(){
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->imGuiInfos();
                }
            }

            void imGuiSettings(){
                for(int i = 0; i < drawables.size(); i++){
                    drawables[i]->imGuiSettings();
                }
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