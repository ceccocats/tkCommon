#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/VectorData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Image : public Drawable {

        private:
            std::vector<std::mutex*>                  mutex;
            std::vector<tk::data::ImageData*>        images;
            std::vector<tk::gui::Texture<uint8_t>*>  textures; 

            std::vector<bool>     new_ref_data;
            std::vector<bool>     initted;
            std::vector<uint32_t> counter;

            bool drw_has_reference = false;

        public:
            Image(int n, std::string name);
            Image(std::string name, int n, ...);
            ~Image();

            void onInit(tk::gui::Viewer *viewer);
            bool isAsyncedCopied(int idx);
            void updateRef(tk::data::ImageData* img);
            void updateRef(int index, tk::data::ImageData* img);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
        
        private:
            void dataRef(tk::gui::Viewer *viewer);
            void dataUpd(tk::gui::Viewer *viewer);
	};
}}