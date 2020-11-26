#pragma once
#include "tkCommon/gui/shader/axisPlot.h"
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/linesMonocolor.h"
#include "tkCommon/gui/shader/circle.h"

#include "tkCommon/math/Vec.h"



namespace tk{ namespace gui{

	class Plot : public Drawable {
        public:
            enum type_t : int {
                LINE = 0, CIRCLES = 1, AXIS = 2
            };

        private:
            std::string name = "plot";
            type_t type;

            std::string item0 = "Lanes";
            std::string item1 = "Circles";
            std::string item2 = "Axis";
            std::vector<const char*> menu;

            //x y z
            CircularArray<tk::common::Vector3<float>> points;
            
            //roll pitch yaw
            CircularArray<tk::common::Vector3<float>> rots;

            tk::gui::Buffer<float> glData;
            tk::gui::shader::linesMonocolor* shaderLine;
            tk::gui::shader::circle*         shaderCircle;
            tk::gui::shader::axisPlot*       shaderAxis;

            float circleRay;
            int   circleRes;
            float lineW;
            float pointW;
            float poseW;
            float drawSize;

            int value;

            int   nPoints  = 0;
            bool  newPoint = true;

            tk::math::Vec3<float> mins, maxs;

            void updateLimits();
            void updateDatas();

        public:

            tk::gui::Color_t color;

            Plot(std::string name, int maxDataDim, type_t type, float drawSize);
            ~Plot();

            void addPoint(tk::common::Tfpose tf);
            void addPoint(tk::math::Vec3<float> pt);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void imGuiSettings();
            void onClose();

            std::string toString();
	};
}}