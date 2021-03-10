#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/CanData.h"

namespace tk{ namespace gui{

	class Can : public Drawable {

        private:
            uint32_t counter = 0;
            bool updateCan = false;
            std::string name;
            tk::data::CanData* can;
            tk::data::CanData* can_tmp;
            std::stringstream print;

            int n = 0;
            int nmsg = 10;
            std::vector<std::string> msg;

        public:
            Can(std::string name = "can");
            Can(tk::data::CanData* data, std::string name = "can");
            ~Can();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(tk::data::CanData* data);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void imGuiSettings();
            void onClose();

            std::string toString();
	};
}}