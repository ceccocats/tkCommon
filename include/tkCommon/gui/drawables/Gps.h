#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/data/GpsData.h"
#include "tkCommon/gui/shader/circle.h"

namespace tk{ namespace gui{

	class Gps : public DataDrawable {

        private:

            int nPos;
            int lastPos;
            std::vector<tk::gui::shader::circle*> circles;

            tk::common::GeodeticConverter geoConv;

            float lineSize = 2.0f;

            double x = 0;
            double y = 0;
            double z = 0;

        public:
            tk::gui::Color_t        color;

            Gps(const std::string& name = "gps" ,int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
            Gps(tk::data::GpsData* gps, const std::string& name = "gps", int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
            ~Gps();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}