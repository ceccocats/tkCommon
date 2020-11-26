#include "tkCommon/gui/drawables/Axis.h"

tk::gui::Axis::Axis(){

}

tk::gui::Axis::~Axis(){

}

void 
tk::gui::Axis::onInit(tk::gui::Viewer *viewer){
    shader = new tk::gui::shader::axis();
}

void 
tk::gui::Axis::draw(tk::gui::Viewer *viewer){
    tk::gui::shader::axis* shaderAxis = (tk::gui::shader::axis*) shader;
    shaderAxis->draw(viewer->getWidth(),viewer->getHeight());
}

void 
tk::gui::Axis::imGuiInfos(){
    ImGui::Text("3D axis drawable");
}

void 
tk::gui::Axis::onClose(){
    tk::gui::shader::axis* shaderAxis = (tk::gui::shader::axis*) shader;
    shaderAxis->close();
    delete shader;
}

std::string 
tk::gui::Axis::toString(){
    return "axis";
}