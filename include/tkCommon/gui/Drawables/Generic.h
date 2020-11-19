#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/data/GpsImuData.h"
#include "tkCommon/data/SensorData.h"

namespace tk{ namespace gui{

	class Generic : public Drawable {
    private:
        std::map<std::string,tk::gui::Drawable> references;
    
    public:
        Generic(){}

        ~Generic(){}

        void updateRef(const tk::data::SensorData* data, int pos = 0) {
            if(data->header.sensor == tk::data::sensorName::GPS){
                
            }




            this->data = *(dynamic_cast<const tk::data::GpsImuData*>(data));
            std::cout<<this->data<<"\n";
        }

        void draw(tk::gui::Viewer *viewer){
        }

        void imGuiInfos(){
            ImGui::Text("%s","Sensor data consumer for viz purpose");
        }

        std::string toString(){
            return "Consumer";
        }
	};
}}