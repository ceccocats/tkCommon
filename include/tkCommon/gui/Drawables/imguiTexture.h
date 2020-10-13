#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/data/ImageData.h"

namespace tk{ namespace gui{

	class imguiTexture : public Drawable {

        private:
            tk::gui::Texture<uint8_t>   texture; 
            tk::data::ImageData*        image;

        public:

            imguiTexture(tk::data::ImageData* image){
                this->image = image;
            }

            ~imguiTexture(){

            }

            void onInit(tk::gui::Viewer *viewer){
                texture.init(image->width, image->height, image->channels);
            }

            void draw(tk::gui::Viewer *viewer){
                if(image->isChange()){
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

            void onClose(){
                texture.release();
            }
	};
}}