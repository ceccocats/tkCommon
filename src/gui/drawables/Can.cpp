#include "tkCommon/gui/drawables/Can.h"

tk::gui::Can::Can(tk::data::CanData_t* data, std::string name){
    this->data = data;
    this->name = name;
}

tk::gui::Can::~Can(){
}

void 
tk::gui::Can::onInit(tk::gui::Viewer *viewer){
}

void 
tk::gui::Can::updateRef(tk::data::CanData_t* data){
    this->data = data;
    update = true;
}

void 
tk::gui::Can::draw(tk::gui::Viewer *viewer){
    if(data->isChanged(counter) || update){
        update = false;
        data->lockRead();
        print.str("");
        print<<(*data);
        data->unlockRead();
    }
}

void 
tk::gui::Can::imGuiInfos(){
    ImGui::Text("%s",print.str().c_str());
}

void 
tk::gui::Can::onClose(){
}

std::string 
tk::gui::Can::toString(){
    return name;
}