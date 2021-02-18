#include "tkCommon/gui/drawables/Grid.h"

tk::gui::Grid::Grid(float dim, int n){
    this->dim = dim;
    this->n = n;
}

tk::gui::Grid::~Grid(){

}

void 
tk::gui::Grid::onInit(tk::gui::Viewer *viewer){
    shader = new tk::gui::shader::grid();
}

void 
tk::gui::Grid::imGuiSettings(){
    ImGui::SliderFloat("Sector dim",&dim, 1.f,10.0f,"%.1f");
    ImGui::SliderInt("Sector number",&n, 10, 1000,"%.1f");
}

void 
tk::gui::Grid::imGuiInfos(){
    ImGui::Text("3D grid drawable");
}

void 
tk::gui::Grid::draw(tk::gui::Viewer *viewer){
    tk::gui::shader::grid* shaderGrid = (tk::gui::shader::grid*) shader;
    shaderGrid->draw(drwModelView,dim,n);
}

void 
tk::gui::Grid::onClose(){
    tk::gui::shader::grid* shaderGrid = (tk::gui::shader::grid*) shader;
    shaderGrid->close();
    delete shaderGrid;
}

std::string 
tk::gui::Grid::toString(){
    return "grid";
}