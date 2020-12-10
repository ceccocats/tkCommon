#include "tkCommon/gui/drawables/GpsImu.h"

tk::gui::GpsImu::GpsImu(){
    gps = new tk::gui::Gps();
    imu = new tk::gui::Imu();
}

tk::gui::GpsImu::~GpsImu(){

}

void 
tk::gui::GpsImu::updateRef(tk::data::GpsImuData* gpsImu){
    name = gpsImu->header.name;
    gps->updateRef(&gpsImu->gps);
    imu->updateRef(&gpsImu->imu);  
    update = true;
}

void 
tk::gui::GpsImu::onInit(tk::gui::Viewer *viewer){
    gps->onInit(viewer);
    imu->onInit(viewer);  
}

void 
tk::gui::GpsImu::draw(tk::gui::Viewer *viewer){
    gps->draw(viewer);
    imu->draw(viewer);      
    update = gps->update && imu->update;            
}

void 
tk::gui::GpsImu::imGuiSettings(){
    gps->imGuiSettings();
    imu->imGuiSettings();  
}

void 
tk::gui::GpsImu::imGuiInfos(){
    gps->imGuiInfos();
    //imu->imGuiInfos();  
}

void 
tk::gui::GpsImu::onClose(){
    gps->onClose();
    imu->onClose(); 
    delete gps;
    delete imu;
}

std::string 
tk::gui::GpsImu::toString(){
    return name;
}