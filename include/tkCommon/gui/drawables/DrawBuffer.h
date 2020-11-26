#pragma once
#include "tkCommon/gui/drawables/Drawable.h"

namespace tk{ namespace gui{

	class DrawBuffer : public Drawable {

        std::string name;

        public:
            std::vector<tk::gui::Drawable*> drawables;
            
            DrawBuffer();
            DrawBuffer(std::string name);
            DrawBuffer(std::string name, int n, ...);
            ~DrawBuffer();

            void onInit(tk::gui::Viewer *viewer);
            void beforeDraw(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void imGuiSettings();
            void onClose();

            std::string toString();
	};
}}