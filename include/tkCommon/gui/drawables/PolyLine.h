#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"

#include "tkCommon/rt/Lockable.h"
#include <eigen3/Eigen/Dense>

namespace tk{ namespace gui{

    class line : public tk::rt::Lockable{
        public:
            std::vector<Eigen::Vector3f> points;
    };


	class PolyLine : public Drawable {

        private:
            tk::gui::line* line;
            tk::gui::Buffer<float> glData;

            uint32_t counter = 0;
            float lineSize = 1.0f;

            bool update;

        public:
            tk::gui::Color_t        color;

            PolyLine(tk::gui::line* line, std::string name = "PolyLines", tk::gui::Color_t col = tk::gui::color::RED);
            ~PolyLine();

            //void updateRef(tk::gui::line* line);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
	};
}}