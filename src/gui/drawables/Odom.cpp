#include "tkCommon/gui/drawables/Odom.h"


tk::gui::Odom::Odom(std::string name)
{    
    this->name      = name;
}

tk::gui::Odom::~Odom()
{
}

void tk::gui::Odom::imGuiInfos()
{
}

void tk::gui::Odom::onClose()
{
}


void tk::gui::Odom::drawData(tk::gui::Viewer *viewer)
{
}

void tk::gui::Odom::updateData(tk::gui::Viewer *viewer)
{
    tk::data::OdomData* p_data = dynamic_cast<tk::data::OdomData*>(data);
    if(p_data->mode == p_data->QUATERNION)
        tf = tk::common::odom2tf(p_data->pose.x(), p_data->pose.y(), p_data->pose.z(), p_data->angle.x(), p_data->angle.y(), p_data->angle.z(), p_data->angle.w());
    else
        tf = tk::common::odom2tf(p_data->pose.x(), p_data->pose.y(), p_data->pose.z(), p_data->angle.x(), p_data->angle.y(), p_data->angle.z());

}

