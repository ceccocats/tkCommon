#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/geometry.h"

namespace tk{ namespace gui{

	class Prisms : public Drawable {
        private:
            tk::common::Prisms prisms;
            tk::common::Prisms* ref;
            uint32_t counter = 0;

            bool update;
        public:

            tk::gui::Color_t color;

            Prisms(tk::gui::Color_t color = tk::gui::color::RED);
            Prisms(tk::common::Prisms& prisms,tk::gui::Color_t color = tk::gui::color::RED);
            ~Prisms();

            void updateData(tk::common::Prisms& prisms);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void imGuiSettings();
            void onClose();
	};
}}