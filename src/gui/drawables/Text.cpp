#include "tkCommon/gui/drawables/Text.h"

tk::gui::Text::Text(std::string text_str){
    this->text_str = text_str;
}

tk::gui::Text::~Text(){

}

void 
tk::gui::Text::setText(std::string t) {
    text_str = t;
}


void 
tk::gui::Text::onInit(tk::gui::Viewer *viewer){
    shader = new tk::gui::shader::text();
    tk::gui::shader::text* shaderText = (tk::gui::shader::text*) shader;
    shaderText->init();
}

void 
tk::gui::Text::draw(tk::gui::Viewer *viewer){
    tk::gui::shader::text* shaderText = (tk::gui::shader::text*) shader;
    shaderText->draw(drwModelView,text_str, text_height, text_color);
}

void 
tk::gui::Text::imGuiInfos(){
    ImGui::Text("Text drawable");
}

void 
tk::gui::Text::onClose(){
    tk::gui::shader::text* shaderText = (tk::gui::shader::text*) shader;
    shaderText->close();
    delete shader;
}

std::string 
tk::gui::Text::toString(){
    return "Text";
}