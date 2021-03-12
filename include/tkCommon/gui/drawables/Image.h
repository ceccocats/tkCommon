#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/VectorData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Image : public DataDrawable {

        private:
            tk::gui::TextureGeneric* texture = nullptr;
            int textureType = -1;

            std::string imguiName;

        public:
            Image(std::string name, std::string imguiName = "", tk::data::SensorData* img = nullptr);
            Image(std::string name, tk::data::SensorData* img = nullptr);
            ~Image();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}