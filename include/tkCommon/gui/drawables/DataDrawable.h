#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/SensorData.h"


namespace tk{ namespace gui{

class DataDrawable : public Drawable {
    public:

        void draw(tk::gui::Viewer *viewer) final;

        virtual void updateRef(tk::data::SensorData* data) final;
        virtual bool synched() final;

    protected:

        void forceUpdate();

        virtual void drawData(tk::gui::Viewer *viewer) = 0; 
        virtual void updateData(tk::gui::Viewer *viewer) = 0; 

        tk::data::SensorData* data = nullptr;
        uint32_t counter = 0;

        std::stringstream print;

    private:

        std::mutex ref_mutex;
        bool new_ref_data = false;

        bool drw_has_reference = true;
	};
}}