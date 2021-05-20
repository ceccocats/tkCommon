#include "tkCommon/gui/drawables/Imu.h"

tk::gui::Imu::Imu(const std::string& name, float delta_ts){ 
    this->delta_ts      = delta_ts;
    this->name          = name;
}

tk::gui::Imu::Imu(tk::data::ImuData* imu, const std::string& name, float delta_ts) : Imu(name, delta_ts){
    this->data = imu;
}

tk::gui::Imu::~Imu(){

}

void 
tk::gui::Imu::updateData(tk::gui::Viewer *viewer){
    tk::data::ImuData* imu = (tk::data::ImuData*)data;
    print.str("");
    print<<(*imu);
    if(prec == 0)
        prec = imu->header.stamp;
    t += float(imu->header.stamp - prec) * 1e-6;
    accX.AddPoint(t, imu->acc.x());
    accY.AddPoint(t, imu->acc.y());
    accZ.AddPoint(t, imu->acc.z());

    roll.AddPoint(t, imu->angle.x());
    pitch.AddPoint(t, imu->angle.y());
    yaw.AddPoint(t, imu->angle.z());

    prec = imu->header.stamp;
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
    if (ImPlot::BeginPlot("##Scrolling0", NULL, NULL, ImVec2(-1,150), 0, rt_axis | ImPlotAxisFlags_Time, rt_axis)) {
        //ImPlot::FitNextPlotAxes(false, true, false, false);
        ImPlot::PlotLine("accX", &accX.Data[0].x, &accX.Data[0].y, accX.Data.size(), accX.Offset, 2*sizeof(float));
        ImPlot::PlotLine("accY", &accY.Data[0].x, &accY.Data[0].y, accY.Data.size(), accY.Offset, 2*sizeof(float));
        ImPlot::PlotLine("accZ", &accZ.Data[0].x, &accZ.Data[0].y, accZ.Data.size(), accZ.Offset, 2*sizeof(float));
        ImPlot::EndPlot();
    }
    ImPlot::SetNextPlotLimitsX(t - delta_ts, t, ImGuiCond_Always);
    ImPlot::SetNextPlotLimitsY(-M_PI, +M_PI);
    if (ImPlot::BeginPlot("##Scrolling1", NULL, NULL, ImVec2(-1,150), 0, rt_axis | ImPlotAxisFlags_Time, rt_axis)) {
        //ImPlot::FitNextPlotAxes(false, true, false, false);
        ImPlot::PlotLine("roll", &roll.Data[0].x, &roll.Data[0].y, roll.Data.size(), roll.Offset, 2*sizeof(float));
        ImPlot::PlotLine("pitch", &pitch.Data[0].x, &pitch.Data[0].y, pitch.Data.size(), pitch.Offset, 2*sizeof(float));
        ImPlot::PlotLine("yaw", &yaw.Data[0].x, &yaw.Data[0].y, yaw.Data.size(), yaw.Offset, 2*sizeof(float));
        ImPlot::EndPlot();
    }
    ImGui::Text("%s",print.str().c_str());
}

