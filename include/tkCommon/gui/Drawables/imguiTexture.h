#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/data/ImageData.h"

namespace tk{ namespace gui{

	class imguiTexture : public Drawable {

        private:
            tk::gui::Texture<uint8_t>   texture; 
            tk::data::ImageData*        image;

            bool update = false;

        public:

            imguiTexture(tk::data::ImageData* image){
                this->image = image;
            }

            ~imguiTexture(){

            }

            void onInit(tk::gui::Viewer *viewer){
                texture.init(image->width, image->height, image->channels);
            }

            void updateRef(tk::data::ImageData* image){
                this->image = image;   
                update = true;
            }

            void draw(tk::gui::Viewer *viewer){
                if(image->isChanged() || update){
                    image->lock();
                    texture.setData(image->data.data_h);
                    image->unlockRead();
                }
                ImGui::Begin(image->header.name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
                int imgX = ImGui::GetWindowSize().x-20;
                int imgY = ImGui::GetWindowSize().y-35;
                ImGui::Image((void*)(intptr_t)texture.id(), ImVec2(imgX, imgY));
                ImGui::End();
            }

            void imGuiInfos(){
                std::stringstream print;
                print<<(*image);
                ImGui::Text("%s",print.str().c_str());
                print.clear();
            }

            void onClose(){
                texture.release();
            }
	};
}}