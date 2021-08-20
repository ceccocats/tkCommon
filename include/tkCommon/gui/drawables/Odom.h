#pragma once
#include "tkCommon/gui/drawables/Image.h"
#include "tkCommon/data/OdomData.h"

#include <cstdarg>

namespace tk{ namespace gui{

	class Odom : public DataDrawable {

        public:
            Odom(std::string name);
            ~Odom();
            
            void imGuiInfos();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}