#include "tkCommon/gui/drawables/Can.h"

tk::gui::Can::Can(std::string name){
    this->name    = name;
    this->initted = false;
    msg.resize(nmsg);
}

tk::gui::Can::Can(tk::data::CanData* data, std::string name){
    this->data    = data;
    this->name    = name;
    this->initted = true;
    msg.resize(nmsg);
}

tk::gui::Can::~Can(){
}

void 
tk::gui::Can::onInit(tk::gui::Viewer *viewer){
}

void 
tk::gui::Can::updateRef(tk::data::CanData* data){
    this->data = data;
    update = initted = true;
}

void 
tk::gui::Can::draw(tk::gui::Viewer *viewer){
    if(initted){
        if(data->isChanged(counter) || update){
            update = false;
            data->lockRead();
            print.str("");
            print<<(*data);

            msg[n] = print.str();
            n = (n+1) % nmsg;

            data->unlockRead();
        }
    }
}

void 
tk::gui::Can::imGuiInfos(){
    for(int i = 0; i < msg.size(); i++){
        ImGui::Text("%s",msg[i].c_str());
    }
}

void 
tk::gui::Can::imGuiSettings(){
    if(ImGui::SliderInt("Last messages",&nmsg,1,40)){
        n = 0;
        msg.resize(nmsg);
    }
}

void 
tk::gui::Can::onClose(){
}

std::string 
tk::gui::Can::toString(){
    return name;
}