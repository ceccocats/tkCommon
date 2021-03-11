#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/SensorData.h"


namespace tk{ namespace gui{

class DataDrawable : public Drawable {
    public:

        void draw(tk::gui::Viewer *viewer) final;

        virtual void updateRef(tk::data::SensorData* data) final;
        virtual bool isAsyncedCopied(int idx = 0) final;

        ~DataDrawable();

    protected:

        void init(int n = 1);
        void forceUpdate();

        virtual void drawData(tk::gui::Viewer *viewer) = 0; 
        virtual void updateData(int i,tk::gui::Viewer *viewer) = 0; 

        std::vector<tk::data::SensorData*> data;
        std::vector<uint32_t> counter;

        std::stringstream print;

    private:

        std::vector<std::mutex*> ref_mutex;
        std::vector<bool> new_ref_data;

        bool drw_has_reference = true;
	};
}}