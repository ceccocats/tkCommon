#include "tkCommon/gui/drawables/Imu.h"

tk::gui::Imu::Imu(){ 
    this->accHistory = 10.0f;
}

tk::gui::Imu::Imu(tk::data::ImuData* imu){
    this->imu = imu;  
    this->accHistory = 10.0f;    
    this->updateImu = true;
}

tk::gui::Imu::~Imu(){

}

void 
tk::gui::Imu::updateRef(tk::data::ImuData* imu){
    mtxUpdate.lock();
    this->imu_tmp = imu;
    update = true;
    mtxUpdate.unlock();
}

void 
tk::gui::Imu::onInit(tk::gui::Viewer *viewer){
}

void 
tk::gui::Imu::draw(tk::gui::Viewer *viewer){
    if(update){
        mtxUpdate.lock();
        update = false;
        imu = imu_tmp;
        mtxUpdate.unlock();
        updateImu = true;
    }

    if(imu == nullptr){
        return;
    }

    if(imu->isChanged(counter)){
        updateImu = true;
    }

    if(updateImu){
        updateImu = false;

        imu->lockRead();
        name = imu->header.name;
        print.str("");
        print<<(*imu);

        if(prec == 0){
            prec = imu->header.stamp;
        }

        t += float(imu->header.stamp - prec) * 1e-6;
        
        accX.AddPoint(t, imu->acc.x());
        accY.AddPoint(t, imu->acc.y());
        accZ.AddPoint(t, imu->acc.z());
        
        prec = imu->header.stamp;

        imu->unlockRead();
    }
}

void 
tk::gui::Imu::imGuiSettings(){
    ImGui::SliderFloat("Acc history",&accHistory,1,30,"%.1f s");
}

void 
tk::gui::Imu::imGuiInfos() {    

    static ImPlotAxisFlags rt_axis = ImPlotAxisFlags_NoTickLabels;
    ImPlot::SetNextPlotLimitsX(t - accHistory, t, ImGuiCond_Always);
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

void 
tk::gui::Imu::onClose(){
}

std::string 
tk::gui::Imu::toString(){
    return name;
}
