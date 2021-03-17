#pragma once
#include "tkCommon/gui/drawables/Image.h"
#include "tkCommon/data/StereoData.h"
#include "tkCommon/data/VectorData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Stereo : public DataDrawable {

        private:
            std::vector<tk::gui::Image*>  images;

            bool has_depth;
            bool has_rgb;

            std::string imguiName;

        public:
            Stereo(std::string name);
            ~Stereo();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(tk::data::StereoData* img);
            void imGuiInfos();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}