#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/utils/Texture.h"

#include "tkCommon/utils.h"

#include "tkCommon/data/ImageData.h"
#include "tkCommon/data/CalibData.h"

namespace tk{namespace gui{

class CameraAR : public tk::gui::Drawable, public tk::rt::Lockable {

public:

    tk::data::ImageData img;
    tk::gui::Texture<uint8_t>* texture; 
    tk::gui::Texture<uint8_t>* im_texture;

    tk::data::CalibData calib;

    tk::common::Tfpose view;

    std::string name;

    std::vector< tk::gui::Drawable * > drawables;

    int w,h,c;

    float z_near = 1.0, z_far = 10000.0;

    tk::gui::Camera camera;

    bool save = false;

    uint32_t counter;

    CameraAR(tk::data::CalibData &calib, int channels = 4, std::string name = "AR");
    
    void updateImage(tk::data::ImageData &im);
    void onInit(tk::gui::Viewer *viewer);
    void imGuiSettings();
    void draw(tk::gui::Viewer *viewer);

};
}}