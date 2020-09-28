#pragma once

#include "tkCommon/data/gen/CloudData_gen.h"

namespace tk { namespace data {

    class CloudData : public CloudData_gen{
    
    private:
        tk::gui::Buffer<float> pointcloud;        
    
    public:

        void onInit(tk::gui::Viewer *viewer){
            shader = new tk::gui::shader::pointcloud4f();
            shader->init();
            pointcloud.init();
        }

        void onAdd(tk::gui::Viewer *viewer){
            pointcloud.setData(points.data_h,points.size());
        }

        void draw(tk::gui::Viewer *viewer){
            shader->draw(&pointcloud, points.cols());			
        }

        void onClose(){
            shader->close();
            pointcloud.release();

            delete [] shader;
        }
    };
}}