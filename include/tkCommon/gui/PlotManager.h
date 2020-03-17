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
                LINE, CIRCLES
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
            } conf;

            CircularArray<tk::common::Vector3<float>> points;
        };

        std::map<std::string, plot_t> plots;


    public:
        void addPlot(std::string id, plot_t::conf_t &pconf) {
            plots[id].conf = pconf;
            plots[id].points.setDim(pconf.maxPoints);
            plots[id].points.clear();
        }

        void addLinePlot(std::string id, Color_t color = tk::gui::color::WHITE, int maxpts = 10000, float lineW = 1) {
            plot_t plot;
            plot.conf.type = plot_t::plottype_t::LINE;
            plot.conf.color = color;
            plot.conf.lineW = lineW;
            plot.conf.maxPoints = maxpts;
            addPlot(id, plot.conf);
        }

        void addCirclePlot(std::string id, Color_t color = tk::gui::color::WHITE, int maxpts = 10000, float circleRay  = 0.5, float lineW = 1, int circleRes = 20) {
            plot_t plot;
            plot.conf.type = plot_t::plottype_t::CIRCLES;
            plot.conf.color = color;
            plot.conf.lineW = lineW;
            plot.conf.circleRay = circleRay;
            plot.conf.circleRes = circleRes;
            plot.conf.maxPoints = maxpts;
            addPlot(id, plot.conf);
        }

        bool plotExist(std::string id) {
            // key alread present
            return plots.find(id) != plots.end();
        }

        void addPoint(std::string id, tk::common::Vector3<float> pt) {
            plots[id].points.add(pt);
        }

        void drawPlots() {
        }

        void drawLegend() {
        }

        plot_t *getPlot(std::string id) {
            if(plotExist(id))
                return &plots[id];
            return nullptr;
        }
    };
}} // namespace name
