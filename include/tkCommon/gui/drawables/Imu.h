#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/ImuData.h"

namespace tk{ namespace gui{

    // utility structure for realtime plot
    struct ScrollingBuffer {
        int MaxSize;
        int Offset;
        ImVector<ImVec2> Data;
        ScrollingBuffer() {
            MaxSize = 2000;
            Offset  = 0;
            Data.reserve(MaxSize);
        }
        void AddPoint(float x, float y) {
            if (Data.size() < MaxSize)
                Data.push_back(ImVec2(x,y));
            else {
                Data[Offset] = ImVec2(x,y);
                Offset =  (Offset + 1) % MaxSize;
            }
        }
        void Erase() {
            if (Data.size() > 0) {
                Data.shrink(0);
                Offset  = 0;
            }
        }
    };

	class Imu : public Drawable {

        private:
            tk::data::ImuData* imu;
            uint32_t counter = 0;

            int nPos;
            bool initted = false;


            std::string name = "";
            std::stringstream print;

            timeStamp_t prec = 0;
            float t = 0;
            ScrollingBuffer accX, accY, accZ;

            float accHistory;

        public:

            Imu();
            Imu(tk::data::ImuData* imu);
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