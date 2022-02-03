#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/VectorData.h"

namespace tk{ namespace gui{

	class Image : public DataDrawable {

        protected:
            tk::gui::TextureGeneric* texture = nullptr;
            int textureType = -1;

            std::string imguiName;

        public:
            Image(std::string name, std::string imguiName = "", tk::data::SensorData* img = nullptr);
            ~Image();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
            void updateData(tk::gui::Viewer *viewer);
        
        protected:
            void drawData(tk::gui::Viewer *viewer);
            
	};
}}