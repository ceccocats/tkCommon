#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/grid.h"

namespace tk{ namespace gui{

	class Grid : public Drawable {

        public:

            float   dim;
            int     n;

            Grid(float dim = 1.0, int n = 50);
            ~Grid();
            
            void onInit(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void draw(tk::gui::Viewer *viewer);
            void onClose();

            std::string toString();
	};
}}