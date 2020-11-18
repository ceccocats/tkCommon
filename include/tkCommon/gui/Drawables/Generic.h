#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/data/GpsImuData.h"

namespace tk{ namespace gui{

	class Generic : public Drawable {
    
    public:
        tk::data::GpsImuData data;
        Generic(){}

        ~Generic(){}

        void updateRef(const tk::data::SensorData* data, int pos = 0) {
            this->data = *(dynamic_cast<const tk::data::GpsImuData*>(data));
            std::cout<<this->data<<"\n";
        }

        void onInit(tk::gui::Viewer *viewer){
        }

        void draw(tk::gui::Viewer *viewer){
        }

        void imGuiSettings(){
        }

        void imGuiInfos(){
            std::stringstream print;
            print<<(data);
            ImGui::Text("%s",print.str().c_str());
            print.clear();
        }

        void onClose(){
        }

        std::string toString(){
            return "Addio";
        }
	};
}}