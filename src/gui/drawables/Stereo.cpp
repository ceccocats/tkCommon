#include "tkCommon/gui/drawables/Stereo.h"


tk::gui::Stereo::Stereo(std::string name){
    
    this->imguiName = name;
    this->name = name;
    this->has_depth = has_depth;
    this->has_rgb = has_rgb;

    int n = 4;

    images.resize(n);
    for (int i = 0; i < n ; ++i){
        images[i] = new tk::gui::Image(name);
    }

    tkDBG("Created with "<< n <<" Images\n");

}

tk::gui::Stereo::~Stereo(){

    for (int i = 0; i < images.size() ; ++i){
        delete images[i];
    }

}


void tk::gui::Stereo::onInit(tk::gui::Viewer *viewer){
    for (int i = 0; i < images.size() ; ++i){
        viewer->add(images[i]);
    }
}

void tk::gui::Stereo::updateRef(tk::data::StereoData* img){


}

void tk::gui::Stereo::imGuiInfos(){
}

void tk::gui::Stereo::onClose(){
}


void tk::gui::Stereo::drawData(tk::gui::Viewer *viewer){
}

void tk::gui::Stereo::updateData(tk::gui::Viewer *viewer){

    tk::data::StereoData *img = (tk::data::StereoData *)data;

    images[0]->updateRef(&img->left);
    images[1]->updateRef(&img->right);

    if(img->color.data.size() > 0){
        images[2]->updateRef(&img->color);
    }
    if(img->depth.data.size() > 0){
        images[3]->updateRef(&img->depth);
    }

}

