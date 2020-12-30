#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/text.h"

namespace tk{ namespace gui{

	class Text : public Drawable {
        private:
            std::string text_str;
            std::mutex text_mutex;

        public:
            float text_height = 1.0;
            tk::gui::Color_t text_color = tk::gui::color::WHITE;

             Text(std::string text_str);
            ~Text();
            
            void setText(std::string t);

            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiInfos();
            void onClose();
            std::string toString();
	};
}}