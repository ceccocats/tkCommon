#include "tkCommon/gui/drawables/Wrapper.h"

template <>
void
tk::gui::Wrapper<tk::data::GpsImuData>::init(){
    drawable = new tk::gui::Gps();
    tk::gui::Viewer *viewer = tk::gui::Viewer::getInstance();
    viewer->add(drawable);
}

template <>
void
tk::gui::Wrapper<tk::data::GpsImuData>::updateRef(const tk::data::SensorData* data){
    auto d  = (tk::data::GpsImuData*) data;
    auto dw = (tk::gui::Gps*) drawable;
    dw->updateRef(&d->gps);
}