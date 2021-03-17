#pragma once

#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/sensor/LogManager.h"


namespace tk { namespace gui {

    class ReplayInfo : public Drawable {
    public:
        ReplayInfo(tk::sensors::LogManager* manager);

        void imGuiSettings();
        void imGuiInfos();
        void onClose();
        
    private:
        tk::sensors::LogManager* manager = nullptr;
        float speed = 1.0f;
    };
}}