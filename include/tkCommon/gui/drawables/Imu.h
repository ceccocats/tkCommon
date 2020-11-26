#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/ImuData.h"

namespace tk{ namespace gui{

	class Imu : public Drawable {

        private:
            tk::data::ImuData* imu;
            uint32_t counter = 0;

            int nPos;
            bool initted = false;


            std::string name = "";
            std::stringstream print;

            const int maxDataDim = 200;
            int prec = 0;
            CircularArray<float> time;
            CircularArray<float> accX;
            CircularArray<float> accY;
            CircularArray<float> accZ;

        public:

            Imu(int nPos = 10);
            Imu(tk::data::ImuData* imu, int nPos = 10);
            ~Imu();

            void updateRef(tk::data::ImuData* imu);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
            
            std::string toString();
	};
}}