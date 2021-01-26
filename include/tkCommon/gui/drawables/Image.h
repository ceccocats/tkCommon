#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/VectorData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Image : public Drawable {

        private:
            std::vector<tk::gui::Texture<uint8_t>*>  textures; 
            std::vector<tk::data::ImageData*>       images;
            std::vector<bool>   updates;
            std::vector<bool>   ready;
            std::vector<uint32_t> counter;


            std::string name;

        public:
            Image(int n, std::string name);
            Image(std::string name, int n, ...);
            ~Image();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(int index, tk::data::ImageData* img);
            void updateRef(tk::data::VectorData<tk::data::ImageData> *vecImg);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();

            std::string toString();
	};
}}