#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/grid.h"

namespace tk{ namespace gui{

	class Grid : public Drawable {

        public:

            float   dim;
            int     n;

            Grid(float dim = 1.0, int n = 50){
                this->dim = dim;
                this->n = n;
            }

            ~Grid(){

            }
            
            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::grid();
            }

            void imGuiSettings(){
                ImGui::SliderFloat("Sector dim",&dim, 1.f,10.0f,"%.1f");
                ImGui::SliderInt("Sector number",&n, 10, 1000,"%.1f");
            }

            void imGuiInfos(){
                ImGui::Text("3D grid drawable");
            }

            void draw(tk::gui::Viewer *viewer){
                tk::gui::shader::grid* shaderGrid = (tk::gui::shader::grid*) shader;
                shaderGrid->draw(dim,n);
            }

            void onClose(){
                tk::gui::shader::grid* shaderGrid = (tk::gui::shader::grid*) shader;
                shaderGrid->close();
                delete shaderGrid;
            }

            std::string toString(){
                return "grid";
            }
	};
}}