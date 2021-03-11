#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/VectorData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Image : public DataDrawable {

        private:
            std::vector<tk::gui::Texture<uint8_t>*>  textures; 
            std::vector<bool>     initted;

        public:
            Image(int n, std::string name);
            Image(std::string name, int n, ...);
            ~Image();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(int index, tk::data::ImageData* img);
            void imGuiInfos();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(int i, tk::gui::Viewer *viewer);
	};
}}