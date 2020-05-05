#include "tkCommon/gui/PlotManager.h"

using namespace tk::gui;

void PlotManager::addPlot(std::string id, plot_t::conf_t &pconf) {
    plots[id].conf = pconf;
    plots[id].points.setDim(pconf.maxPoints);
    plots[id].points.clear();
}

void PlotManager::addLinePlot(std::string id, Color_t color, int maxpts, float lineW) {
    plot_t plot;
    plot.conf.type = plot_t::plottype_t::LINE;
    plot.conf.color = color;
    plot.conf.lineW = lineW;
    plot.conf.maxPoints = maxpts;
    addPlot(id, plot.conf);
}

void PlotManager::addCirclePlot(std::string id, Color_t color, int maxpts, float circleRay, float lineW, int circleRes) {
    plot_t plot;
    plot.conf.type = plot_t::plottype_t::CIRCLES;
    plot.conf.color = color;
    plot.conf.lineW = lineW;
    plot.conf.circleRay = circleRay;
    plot.conf.circleRes = circleRes;
    plot.conf.maxPoints = maxpts;
    addPlot(id, plot.conf);
}

void PlotManager::addPosesPlot(std::string id, Color_t color, int maxpts, float poseW) {
    plot_t plot;
    plot.conf.type = plot_t::plottype_t::POSES;
    plot.conf.color = color;
    plot.conf.poseW = poseW;
    plot.conf.maxPoints = maxpts;
    addPlot(id, plot.conf);
}

bool PlotManager::plotExist(std::string id) {
    // key alread present
    return plots.find(id) != plots.end();
}

void PlotManager::addPoint(std::string id, tk::common::Vector3<float> pt) {
    plots[id].points.add(tk::common::odom2tf(pt.x, pt.y, pt.z, 0, 0, 0));
    plots[id].updateLimits();
}

void PlotManager::addPoint(std::string id, tk::common::Tfpose pt) {
    plots[id].points.add(pt);
    plots[id].updateLimits();
}

void PlotManager::addPoints(std::string id, std::vector<tk::common::Vector3<float>> pts) {
    for(int i=0; i<pts.size(); i++) {
        plots[id].points.add(tk::common::odom2tf(pts[i].x, pts[i].y, pts[i].z, 0, 0, 0));
        plots[id].updateLimits();
    }
}

void PlotManager::drawPlots() {
    // Iterate through all elements in std::map
    std::map<std::string, plot_t>::iterator it = plots.begin();
    while(it != plots.end()) {
        plot_t *p = &it->second;
        if(p->conf.show && p->conf.is2d == false) {
            tk::gui::Viewer::tkSetColor(p->conf.color);
            glLineWidth(p->conf.lineW);
            glPointSize(p->conf.pointW);

            if (p->conf.type == plot_t::plottype_t::LINE) {
                for (int i = 0; i < p->points.size() - 1; i++) {
                    tk::gui::Viewer::tkDrawLine(tk::common::tf2pose(p->points.head(i)), tk::common::tf2pose(p->points.head(i + 1)));
                }

            } else if (p->conf.type == plot_t::plottype_t::CIRCLES) {
                for (int i = 0; i < p->points.size(); i++) {
                    tk::gui::Viewer::tkDrawCircle(tk::common::tf2pose(p->points.head(i)), p->conf.circleRay, p->conf.circleRes);
                }
            } else if (p->conf.type == plot_t::plottype_t::POSES) {
                for (int i = 0; i < p->points.size(); i++) {
                    tk::gui::Viewer::tkDrawArrow(
                        tk::common::tf2pose(p->points.head(i)), 
                        tk::common::tf2rot(p->points.head(i)).z, 
                        p->conf.poseW);
                }
            }

            glLineWidth(1);
            glPointSize(1);
        }
        it++;
    }
}

void PlotManager::drawLegend() {

    if(plots.size() > 0) {
        ImGui::SetNextWindowCollapsed(true, ImGuiCond_FirstUseEver);
        ImGui::Begin("plots");

        std::string show2d = "";
        std::map<std::string, plot_t>::iterator it = plots.begin();
        while(it != plots.end()) {
            plot_t *p = &it->second;
            tk::gui::Color_t col = p->conf.color;

            std::string type_str;
            if(p->conf.type == plot_t::plottype_t::LINE)
                type_str = "-";
            else if(p->conf.type == plot_t::plottype_t::CIRCLES)
                type_str = "o";

            if(p->conf.is2d)
                type_str += " 2d";

            ImGui::Checkbox(it->first.c_str(), &p->conf.show);
            ImGui::SameLine(0, 10);
            ImGui::TextColored(ImVec4(float_t(col.r)/255.0f, float_t(col.g)/255.0f, float_t(col.b)/255.0f, float_t(col.a)/255.0f), "%s", type_str.c_str());

            if(p->conf.show && p->conf.is2d)
                show2d = it->first;
            it++;
        }
        ImGui::End();


        if(plotExist(show2d)) {
            plot_t *p = &plots[show2d];

            // draw 2d plot inside the viewport (setted by the viewer)
            tk::gui::Viewer::tkSetColor(tk::gui::color::BLACK, 0.5);
            tk::gui::Viewer::tkDrawRectangle({0,0,0}, {2,2,0}, true);
            tk::gui::Viewer::tkSetColor(tk::gui::color::WHITE);
            tk::gui::Viewer::tkDrawRectangle({0,0,-0.1}, {2,2,0}, false);

            // plot name
            tk::gui::Viewer::tkSetColor(tk::gui::color::YELLOW);
            tk::gui::Viewer::tkDrawText(show2d, {-0.95,+0.85,-0.2}, {0,0,0}, {0.15, 0.15, 0.1});
            // limits
            tk::gui::Viewer::tkSetColor(tk::gui::color::WHITE);
            tk::gui::Viewer::tkDrawText(std::to_string(p->conf.maxs.y) , {0,+0.9,-0.2}, {0,0,0}, {0.1, 0.1, 0.1});
            tk::gui::Viewer::tkDrawText(std::to_string(p->conf.mins.y) , {0,-0.96,-0.2}, {0,0,0}, {0.1, 0.1, 0.1});
            tk::gui::Viewer::tkDrawText(std::to_string(p->conf.maxs.x) , {+0.96,0,-0.2}, {0,0,M_PI/2}, {0.1, 0.1, 0.1});
            tk::gui::Viewer::tkDrawText(std::to_string(p->conf.mins.x) , {-0.9,0,-0.2}, {0,0,M_PI/2}, {0.1, 0.1, 0.1});
            // axes
            if(p->conf.mins.x != 0 || p->conf.mins.y !=0 || p->conf.maxs.x != 0 || p->conf.maxs.y != 0) {
                tk::common::Vector3<float> centerPt = {0,0,0};
                centerPt.x = (centerPt.x - p->conf.mins.x) / (p->conf.maxs.x - p->conf.mins.x) -1.0f;
                centerPt.y = (centerPt.y - p->conf.mins.y) / (p->conf.maxs.y - p->conf.mins.y) -1.0f;
                tk::gui::Viewer::tkDrawLine({-1,centerPt.y,-0.25}, {+1,centerPt.y,-0.25});
                tk::gui::Viewer::tkDrawLine({centerPt.x,-1,-0.25}, {centerPt.x,+1,-0.25});
            }

            // calculate ray
            tk::gui::Viewer::tkSetColor(p->conf.color);
            float rx = p->conf.circleRay / (p->conf.maxs.x - p->conf.mins.x);
            float ry = p->conf.circleRay / (p->conf.maxs.y - p->conf.mins.y);
            float r = rx < ry ? rx : ry;

            for(int i=0; i<p->points.size(); i++) {
                tk::common::Vector3<float> pt = tk::common::tf2pose(p->points.head(i));
                pt.x = (pt.x - p->conf.mins.x) / (p->conf.maxs.x - p->conf.mins.x) -1.0f;
                pt.y = (pt.y - p->conf.mins.y) / (p->conf.maxs.y - p->conf.mins.y) -1.0f;
                pt.z = -0.21;
                tk::gui::Viewer::tkDrawCircle(pt, r, 20);
            }

        }

    }


}

PlotManager::plot_t *PlotManager::getPlot(std::string id) {
    if(plotExist(id))
        return &plots[id];
    return nullptr;
}

void PlotManager::set2d(std::string id, bool is2d) {
    if(plotExist(id))
        plots[id].conf.is2d = true;
}

