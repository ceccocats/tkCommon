#include "tkCommon/gui/Drawables/Wrapper.h"

template <>
void Wrapper::init<tk::data::GpsImuData>(){
    drawable = new tk::gui::Gps();
}

template <>
void updateRef<tk::data::GpsImuData>(const tk::data::SensorData* data){
    auto d = (tk::data::GpsImuData*) data;
    drawable->updateRef(d.gps);
}