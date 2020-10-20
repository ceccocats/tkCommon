#include "tkCommon/gui/utils/PlotManager.h"

using namespace tk::gui;

void PlotManager::addPlot(std::string id, plot_t::conf_t &pconf) {
    //tkASSERT(id != "");
    if(id == "") {
        return;
    }
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
    //tkASSERT(id != "");
    // key alread present
    return plots.find(id) != plots.end();
}

void PlotManager::addPoint(std::string id, tk::common::Vector3<float> pt) {
    if(plotExist(id)) {
        plots[id].points.add(tk::common::odom2tf(pt.x, pt.y, pt.z, 0, 0, 0));
        plots[id].updateLimits();
    }
}

void PlotManager::addPoint(std::string id, tk::common::Tfpose pt) {
    if(plotExist(id)) {
        plots[id].points.add(pt);
        plots[id].updateLimits();
    }
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
            ImGui::TextColored(ImVec4(col.r(),col.g(),col.b(),col.a()), "%s", type_str.c_str());

            if(p->conf.show && p->conf.is2d)
                show2d = it->first;
            it++;
        }
        ImGui::End();
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

