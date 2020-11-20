#include "tkCommon/gui/drawables/Wrapper.h"

template <>
void
tk::gui::Wrapper<tk::data::GpsImuData>::init(){
    drawable = new tk::gui::Gps();
}

template <>
void
tk::gui::Wrapper<tk::data::GpsImuData>::updateRef(const tk::data::SensorData* data){
    auto d  = (tk::data::GpsImuData*) data;
    auto dw = (tk::gui::Gps*) drawable;
    dw->updateRef(&d->gps);
}