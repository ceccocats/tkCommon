#include "tkCommon/gui/drawables/Stereo.h"


tk::gui::Stereo::Stereo(std::string name)
{    
    this->imguiName = name;
    this->name      = name;
    this->has_depth = true;
    this->has_rgb   = true;

    images.resize(4);
    images[0] = new tk::gui::Image( name + "_left", name );
    images[1] = new tk::gui::Image( name + "_right", name );
    images[2] = new tk::gui::Image( name + "_rgb", name );
    images[3] = new tk::gui::Image( name + "_depth", name );
}

tk::gui::Stereo::~Stereo()
{
    for (int i = 0; i < images.size() ; ++i){
        delete images[i];
    }
}


void tk::gui::Stereo::onInit(tk::gui::Viewer *viewer)
{
    for (int i = 0; i < images.size() ; ++i) {
        viewer->add(images[i]);
    }
}

//void tk::gui::Stereo::updateRef(tk::data::StereoData* img)
//{
//}

void tk::gui::Stereo::imGuiInfos()
{
}

void tk::gui::Stereo::onClose()
{
}


void tk::gui::Stereo::drawData(tk::gui::Viewer *viewer)
{
}

void tk::gui::Stereo::updateData(tk::gui::Viewer *viewer)
{
    auto img = dynamic_cast<tk::data::StereoData*>(data);

    images[0]->data = dynamic_cast<tk::data::SensorData*>(&img->left);
    images[0]->updateData(viewer);

    images[1]->data = dynamic_cast<tk::data::SensorData*>(&img->right);
    images[1]->updateData(viewer);

    if(img->color.data.size() > 0){
        images[2]->data = dynamic_cast<tk::data::SensorData*>(&img->color);
        images[2]->updateData(viewer);
    }
    if(img->depth.data.size() > 0){
        images[3]->data = dynamic_cast<tk::data::SensorData*>(&img->depth);
        images[3]->updateData(viewer);
    }
}

