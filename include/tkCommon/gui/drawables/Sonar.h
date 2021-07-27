#pragma once
#include "tkCommon/gui/drawables/Image.h"
#include "tkCommon/gui/drawables/Imu.h"
#include "tkCommon/data/SonarData.h"

namespace tk{ namespace gui{

	class Sonar : public DataDrawable {

        private:
            tk::gui::Texture<float>* texture = nullptr;
            tk::gui::ScrollingBuffer roll, pitch, yaw;
            float t = 0;
            timeStamp_t prec = 0;
            const float delta_ts = 10.0f;

        public:
            Sonar(std::string name, tk::data::SensorData* sonar = nullptr);
            ~Sonar();

            void onInit(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
            void updateData(tk::gui::Viewer *viewer);
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            
	};
}}