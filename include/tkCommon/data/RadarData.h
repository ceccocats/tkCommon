#pragma once

#include "tkCommon/data/gen/RadarData_gen.h"

namespace tk { namespace data {

    class RadarData : public RadarData_gen {

    private:
        tk::gui::Buffer<float> nearCloud;   
        tk::gui::Buffer<float> farCloud;     
    
    public:

        void onInit(tk::gui::Viewer *viewer){
            shader = new tk::gui::shader::pointcloud4f();
            shader->init();
            nearCloud.init();
            farCloud.init();
        }

        void onAdd(tk::gui::Viewer *viewer){
            nearCloud.setData(near.data_h,near.size());
            farCloud.setData(far.data_h,far.size());
        }

        void draw(tk::gui::Viewer *viewer){
            shader->draw(&nearCloud, points.cols(),tk::gui::color::GREEN);	
            shader->draw(&farCloud, points.cols(),tk::gui::color::CYAN);		
        }

        void onClose(){
            shader->close();
            farCloud.release();
            nearCloud.release();

            delete shader;
        }
    };
}}