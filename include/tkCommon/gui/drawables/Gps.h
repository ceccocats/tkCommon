#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/data/GpsData.h"
#include "tkCommon/gui/shader/circle.h"
#include "tkCommon/projection/Projector.h"

namespace tk{ namespace gui{
	class Gps : public DataDrawable {
        private:
            int                 nPos;
            int                 lastPos;
            float               lineSize = 2.0f;
            static const int    MAX_POSES = 40;
            std::vector<tk::gui::shader::circle*>   circles;
            tk::projection::Projector               *proj;
            
        public:
            tk::gui::Color_t        color;

             Gps(const std::string& name = "gps", tk::projection::ProjectionType prj_type = tk::projection::ProjectionType::UTM, int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
             Gps(tk::data::GpsData* gps, const std::string& name = "gps", tk::projection::ProjectionType prj_type = tk::projection::ProjectionType::UTM, int nPos = 10, tk::gui::Color_t color = tk::gui::color::RED);
            ~Gps() = default;

            void onInit(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();

            void setOrigin(double aOriginLat, double aOriginLon, double aOriginEle);
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}