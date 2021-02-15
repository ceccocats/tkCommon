#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/data/CanData.h"

namespace tk{ namespace gui{

	class Can : public Drawable {

        private:
            uint32_t counter = 0;
            std::string name;
            tk::data::CanData_t* data;
            std::stringstream print;

        public:
            Can(tk::data::CanData_t* data, std::string name = "can");
            ~Can();

            void onInit(tk::gui::Viewer *viewer);
            void updateRef(tk::data::CanData_t* data);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();

            std::string toString();
	};
}}