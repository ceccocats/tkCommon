#pragma once

#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk{ namespace data{

    class ImageData : public ImageData_gen{

    private:
        tk::gui::Texture<uint8_t> texture; 

    public:

        void onInit(tk::gui::Viewer *viewer){
            texture.init(width, height, channels);
        }

        void onAdd(tk::gui::Viewer *viewer){
		    texture.setData(data.data_h);
        }

        void draw(tk::gui::Viewer *viewer){
            ImGui::Begin(header.name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
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