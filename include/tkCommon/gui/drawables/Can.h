#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/CanData.h"

namespace tk{ namespace gui{

	class Can : public Drawable {

        private:
            uint32_t counter = 0;
            bool initted;
            std::string name;
            tk::data::CanData_t* data;
            std::stringstream print;

            int n = 0;
            int nmsg = 10;
            std::vector<std::string> msg;

        public:
            Can(std::string name = "can");
            Can(tk::data::CanData_t* data, std::string name = "can");
            ~Can();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(tk::data::CanData_t* data);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void imGuiSettings();
            void onClose();

            std::string toString();
	};
}}