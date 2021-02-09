#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/data/GpsData.h"
#include "tkCommon/gui/shader/circle.h"

namespace tk{ namespace gui{

	class Gps : public Drawable {

        private:
            tk::data::GpsData* gps;
            uint32_t counter = 0;

            int nPos;
            int lastPos;
            std::vector<tk::gui::shader::circle*> circles;

            tk::common::GeodeticConverter geoConv;

            float lineSize = 2.0f;

            bool initted = false;

            std::string name = "";
            std::stringstream print;

        public:
            tk::gui::Color_t        color;

            Gps(const std::string& name = "gps" ,int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
            Gps(tk::data::GpsData* gps, const std::string& name = "gps", int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
            ~Gps();

            void updateRef(tk::data::GpsData* gps);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
            
            std::string toString();
	};
}}