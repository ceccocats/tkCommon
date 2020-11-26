#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/axis.h"

namespace tk{ namespace gui{

	class Axis : public Drawable {

        public:

            float   dim;
            int     n;

            Axis();
            ~Axis();
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
            std::string toString();
	};
}}