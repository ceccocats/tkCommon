#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/mesh.h"

namespace tk{ namespace gui{

	class Mesh : public Drawable {

        protected:
            std::vector<tk::gui::Buffer<float>> obj;
            std::vector<tk::gui::Color_t> objColors;
            std::string filename;
            float ambientStrength;
            bool useLight;

        public:

            Mesh(std::string filename, float ambientStrength = 0.6f, bool useLight = true);
            ~Mesh();

            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void beforeDraw();
            void onClose();

            std::vector<tk::gui::Buffer<float>> & getBuffer(){return obj;}
            std::vector<tk::gui::Color_t> & getColors(){return objColors;}
            std::vector<tk::gui::Color_t> & setColors(){return objColors;}
            float getAmbientStrengt(){return ambientStrength;}
            void setAmbientStrengt(float strength){ambientStrength = strength;}
            void setLight(bool val){useLight = val;};

            std::string toString();
	};
}}