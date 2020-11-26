#include "tkCommon/gui/drawables/Imu.h"

tk::gui::Imu::Imu(int nPos){ 
    this->nPos = nPos;
    this->initted = false;
    time.setDim(maxDataDim);
    accX.setDim(maxDataDim);
    accY.setDim(maxDataDim);
    accZ.setDim(maxDataDim);
    for(int i = 0; i < maxDataDim; i++){
        time.array[i] = accX.array[i] = 0;
        accY.array[i] = accZ.array[i] = 0;
    }
}

tk::gui::Imu::Imu(tk::data::ImuData* imu, int nPos){
    this->imu = imu;  
    this->nPos = nPos;
    this->initted = true;
    time.setDim(maxDataDim);
    accX.setDim(maxDataDim);
    accY.setDim(maxDataDim);
    accZ.setDim(maxDataDim);
    for(int i = 0; i < maxDataDim; i++){
        time.array[i] = accX.array[i] = 0;
        accY.array[i] = accZ.array[i] = 0;
    }
}

tk::gui::Imu::~Imu(){

}

void 
tk::gui::Imu::updateRef(tk::data::ImuData* imu){
    this->imu = imu;   
    initted = update = true;
}

void 
tk::gui::Imu::onInit(tk::gui::Viewer *viewer){
}

void 
tk::gui::Imu::draw(tk::gui::Viewer *viewer){
    if(initted == true){
        if(imu->isChanged(counter) || update){
            update = false;

            imu->lockRead();
            name = imu->header.name;
            print.str("");
            print<<(*imu);

            if(prec == 0){
                prec = imu->header.stamp-1;
            }
            time.add(imu->header.stamp - prec);
            prec = imu->header.stamp;

            accX.add(imu->acc.x());
            accY.add(imu->acc.y());
            accZ.add(imu->acc.z());

            imu->unlockRead();
        }
    }
}

void 
tk::gui::Imu::imGuiSettings(){
    if(ImGui::SliderInt("Last poses",&nPos,1,maxDataDim)){
        time.setDim(nPos);
        accX.setDim(nPos);
        accY.setDim(nPos);
        accZ.setDim(nPos);
    }
}

void 
tk::gui::Imu::imGuiInfos(){
    ImPlot::BeginPlot("##Rolling", NULL, NULL, ImVec2(-1,150), 0, 
        ImPlotAxisFlags_NoTickLabels, ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_LockMin);
    ImPlot::PlotLine("Acc x", (const float*)&time.array, (const float*)&accX.array, nPos, -3, 3);
    ImPlot::PlotLine("Acc y", (const float*)&time.array, (const float*)&accY.array, nPos, -3, 3);
    ImPlot::PlotLine("Acc z", (const float*)&time.array, (const float*)&accZ.array, nPos, -3, 3);
    ImPlot::EndPlot();
    ImGui::Text("%s",print.str().c_str());
}

void 
tk::gui::Imu::onClose(){
}

std::string 
tk::gui::Imu::toString(){
    return name;
}
