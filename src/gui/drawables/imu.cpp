#include "tkCommon/gui/drawables/Imu.h"

tk::gui::Imu::Imu(const std::string& name, float delta_ts){ 
    this->delta_ts      = delta_ts;
    this->name          = name;
}

tk::gui::Imu::Imu(tk::data::ImuData* imu, const std::string& name, float delta_ts) : Imu(name, delta_ts){
    this->data          = imu;
}

tk::gui::Imu::~Imu(){

}

void 
tk::gui::Imu::updateData(tk::gui::Viewer *viewer){
    print.str("");
    print<<(*data);
    if(prec == 0)
        prec = data->header.stamp;
    t += float(data->header.stamp - prec) * 1e-6;
    accX.AddPoint(t, data->acc.x());
    accY.AddPoint(t, data->acc.y());
    accZ.AddPoint(t, data->acc.z());
    prec = data->header.stamp;
}

void 
tk::gui::Imu::imGuiSettings(){
    ImGui::SliderFloat("Acc history",&delta_ts,1,30,"%.1f s");
}

void 
tk::gui::Imu::imGuiInfos() {    

    static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
    ImPlot::SetNextPlotLimitsX(t - delta_ts, t, ImGuiCond_Always);
    ImPlot::SetNextPlotLimitsY(-10.0f, 10.0f);
    if (ImPlot::BeginPlot("##Scrolling", NULL, NULL, ImVec2(-1,150), 0, rt_axis | ImPlotAxisFlags_Time, rt_axis)) {
        //ImPlot::FitNextPlotAxes(false, true, false, false);
        ImPlot::PlotLine("accX", &accX.Data[0].x, &accX.Data[0].y, accX.Data.size(), accX.Offset, 2*sizeof(float));
        ImPlot::PlotLine("accY", &accY.Data[0].x, &accY.Data[0].y, accY.Data.size(), accY.Offset, 2*sizeof(float));
        ImPlot::PlotLine("accZ", &accZ.Data[0].x, &accZ.Data[0].y, accZ.Data.size(), accZ.Offset, 2*sizeof(float));
        ImPlot::EndPlot();
    }
    ImGui::Text("%s",print.str().c_str());
}

