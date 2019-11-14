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

        void addLinePlot(std::string id, Color_t color = tk::gui::color::WHITE, float lineW = 1) {
            plot_t plot;
            plot.conf.type = plot_t::plottype_t::LINE;
            plot.conf.color = color;
            plot.conf.lineW = lineW;
            addPlot(id, plot.conf);
        }

        void addCirclePlot(std::string id, Color_t color = tk::gui::color::WHITE, float circleRay  = 0.5, float lineW = 1, int circleRes = 20) {
            plot_t plot;
            plot.conf.type = plot_t::plottype_t::CIRCLES;
            plot.conf.color = color;
            plot.conf.lineW = lineW;
            plot.conf.circleRay = circleRay;
            plot.conf.circleRes = circleRes;
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
            // Iterate through all elements in std::map
            std::map<std::string, plot_t>::iterator it = plots.begin();
            while(it != plots.end()) {
                plot_t *p = &it->second;
                if(p->conf.show) {
                    tk::gui::Viewer::tkSetColor(p->conf.color);
                    glLineWidth(p->conf.lineW);
                    glPointSize(p->conf.pointW);

                    if (p->conf.type == plot_t::plottype_t::LINE) {
                        for (int i = 0; i < p->points.size() - 1; i++) {
                            tk::gui::Viewer::tkDrawLine(p->points.head(i), p->points.head(i + 1));
                        }

                    } else if (p->conf.type == plot_t::plottype_t::CIRCLES) {
                        for (int i = 0; i < p->points.size() - 1; i++) {
                            tk::gui::Viewer::tkDrawCircle(p->points.head(i), p->conf.circleRay, p->conf.circleRes);
                        }
                    }

                    glLineWidth(1);
                    glPointSize(1);
                }
                it++;
            }
        }

        void drawLegend() {

            if(plots.size() > 0) {

                ImGui::SetNextWindowPos(ImVec2(ImGui::GetWindowWidth()-50, 10), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
                ImGui::Begin("Legend");

                std::map<std::string, plot_t>::iterator it = plots.begin();
                while(it != plots.end()) {
                    plot_t *p = &it->second;
                    tk::gui::Color_t col = p->conf.color;

                    std::string type_str;
                    if(p->conf.type == plot_t::plottype_t::LINE)
                        type_str = "-";
                    else if(p->conf.type == plot_t::plottype_t::CIRCLES)
                        type_str = "o";

                    ImGui::Checkbox(it->first.c_str(), &p->conf.show);
                    ImGui::SameLine(0, 10);
                    ImGui::TextColored(ImVec4(float_t(col.r)/255.0f, float_t(col.g)/255.0f, float_t(col.b)/255.0f, float_t(col.a)/255.0f), type_str.c_str());

                    it++;
                }
                ImGui::End();
            }
        }
    };
}} // namespace name
