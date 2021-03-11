#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/CanData.h"

namespace tk{ namespace gui{

	class Can : public DataDrawable<tk::data::CanData> {

        private:
            static const int MAX_MSG = 40;
            
            int n_msg = 10;
            CircularArray<std::string> msg;

        public:
            Can(std::string name = "can");
            Can(tk::data::CanData* data, std::string name = "can");
            ~Can();

            void imGuiInfos();
            void imGuiSettings();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer){};
            void updateData(tk::gui::Viewer *viewer);
	};
}}