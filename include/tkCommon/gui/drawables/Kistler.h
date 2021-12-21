#pragma once
#include "tkCommon/gui/drawables/DataDrawable.h"
#include "tkCommon/data/KistlerData.h"

namespace tk{ namespace gui{

	class Kistler : public DataDrawable {

        private:

        public:
            Kistler(std::string name = "kistler");
            Kistler(tk::data::KistlerData* data, std::string name = "kistler");
            ~Kistler();

            void imGuiInfos();
            void imGuiSettings();
            void onClose();
        
        private:
            void drawData(tk::gui::Viewer *viewer);
            void updateData(tk::gui::Viewer *viewer);
	};
}}