#pragma once

#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/RadarData.h"
#include "tkCommon/gui/drawables/Cloud4f.h"

namespace tk { namespace gui {

	class Radar : public DataDrawable{

        private:
            tk::gui::Cloud4f *far_drw, *near_drw;

        public:
            Radar(const std::string& name = "radar");
            Radar(tk::data::RadarData* radar, const std::string& name);
            ~Radar();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            
        private:
            void drawData(tk::gui::Viewer *viewer){};
            void updateData(int i, tk::gui::Viewer *viewer);
	};
}}