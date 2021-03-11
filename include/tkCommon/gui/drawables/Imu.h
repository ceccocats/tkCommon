#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
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

	class Imu : public DataDrawable<tk::data::ImuData> {

        private:
            float t = 0;
            float delta_ts = 0;
            timeStamp_t prec = 0;
            ScrollingBuffer accX, accY, accZ;

        public:
            Imu(const std::string& name = "imu", float delta_ts = 10.0f);
            Imu(tk::data::ImuData* imu, const std::string& name = "imu", float delta_ts = 10.0f);
            ~Imu();

            void imGuiSettings();
            void imGuiInfos();
            
        private:
            void drawData(tk::gui::Viewer *viewer){};
            void updateData(tk::gui::Viewer *viewer);

	};
}}