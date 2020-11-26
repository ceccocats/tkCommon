#pragma once
#include "tkCommon/gui/drawables/Gps.h"
#include "tkCommon/gui/drawables/Imu.h"
#include "tkCommon/data/GpsImuData.h"

namespace tk{ namespace gui{

	class GpsImu : public Drawable {

        private:
            tk::data::GpsImuData* data;

            std::string name = "GpsImu";

        public:
            tk::gui::Gps* gps;
            tk::gui::Imu* imu;

            GpsImu();
            ~GpsImu();

            void updateRef(tk::data::GpsImuData* gpsImu);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();

            std::string toString();
	};
}}