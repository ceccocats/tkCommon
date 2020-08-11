#pragma once
#include <map>
#include "tkCommon/common.h"
#include "tkCommon/utils.h"
#include "tkCommon/gui/Viewer.h"
#include "Viewer.h"

namespace tk { namespace gui {

    class PlotManager {

        struct plot_t {
            enum plottype_t {
                LINE, CIRCLES, POSES
            };

            struct conf_t {
                plottype_t type;
                tk::gui::Color_t color = tk::gui::color::WHITE;
                float circleRay = 0.5f;
                int circleRes = 20;
                float lineW = 1;
                float pointW = 1;
                int maxPoints = 10000;
                bool show = true;
                float poseW = 1;

                bool is2d = false;
                tk::common::Vector3<float> mins, maxs;

            } conf;

            CircularArray<tk::common::Tfpose> points;

            void updateLimits() {
                float x = points.head(0).matrix()(0, 3);
                float y = points.head(0).matrix()(1, 3);
                float z = points.head(0).matrix()(2, 3);

                if(points.size() == 1) {
                    conf.mins = { x, y, z };
                    conf.maxs = { x, y, z };
                }
                if(x < conf.mins.x) conf.mins.x = x;
                if(y < conf.mins.y) conf.mins.y = y;
                if(z < conf.mins.z) conf.mins.z = z;
                if(x > conf.maxs.x) conf.maxs.x = x;
                if(y > conf.maxs.y) conf.maxs.y = y;
                if(z > conf.maxs.z) conf.maxs.z = z;
            }
        };

        std::map<std::string, plot_t> plots;

    public:
        void addPlot(std::string id, plot_t::conf_t &pconf);
        void addLinePlot(std::string id, Color_t color = tk::gui::color::WHITE, int maxpts = 10000, float lineW = 1);
        void addCirclePlot(std::string id, Color_t color = tk::gui::color::WHITE, int maxpts = 10000, float circleRay  = 0.5, float lineW = 1, int circleRes = 20);
        void addPosesPlot(std::string id, Color_t color = tk::gui::color::WHITE, int maxpts = 10000, float poseW = 1);

        void addPoint(std::string id, tk::common::Vector3<float> pt);
        void addPoint(std::string id, tk::common::Tfpose pt);
        void addPoints(std::string id, std::vector<tk::common::Vector3<float>> pts);

        bool plotExist(std::string id);
        plot_t *getPlot(std::string id);
        void set2d(std::string id, bool is2d);

        void drawPlots();
        void drawLegend();
    };
}} // namespace name
