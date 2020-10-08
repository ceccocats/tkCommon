#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/axis.h"

namespace tk{ namespace gui{

	class Axis : public Drawable {

        public:

            float   dim;
            int     n;

            Axis(){

            }

            ~Axis(){

            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::axis();
            }

            void draw(tk::gui::Viewer *viewer){
                tk::gui::shader::axis* shaderAxis = (tk::gui::shader::axis*) shader;
                shaderAxis->draw();
            }

            void onClose(){
                tk::gui::shader::axis* shaderAxis = (tk::gui::shader::axis*) shader;
                shaderAxis->close();
                delete shader;
            }

            std::string toString(){
                return "axis";
            }
	};
}}