#pragma once

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/RadarData.h"
#include "tkCommon/gui/drawables/Cloud4f.h"

namespace tk { namespace gui {

	class Radar : public Drawable {

        private:
            tk::data::RadarData* radar;
            uint32_t counter = 0;
            bool initted = false;

            Cloud4f *far_drw, *near_drw;

            std::string name = "";
            std::stringstream print;
        public:
            Radar();
            Radar(const std::string& name);
            Radar(tk::data::RadarData* radar, const std::string& name);
            ~Radar();

            void updateRef(tk::data::RadarData* radar);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
            
            std::string toString();
	};
}}