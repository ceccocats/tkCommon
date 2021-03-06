#include "tkCommon/gui/drawables/Can.h"

tk::gui::Can::Can(std::string name){
    this->name       = name;
    this->updateCan  = false;
    msg.resize(nmsg);
}

tk::gui::Can::Can(tk::data::CanData_t* data, std::string name){
    this->can        = data;
    this->name       = name;
    this->updateCan  = true;
    msg.resize(nmsg);
}

tk::gui::Can::~Can(){
}

void 
tk::gui::Can::onInit(tk::gui::Viewer *viewer){
}

void 
tk::gui::Can::updateRef(tk::data::CanData_t* data){
    mtxUpdate.lock();
    this->can_tmp = data;
    update = true;
    mtxUpdate.unlock();
}

void 
tk::gui::Can::draw(tk::gui::Viewer *viewer){

    if(update){
        mtxUpdate.lock();
        update = false;
        can = can_tmp;
        mtxUpdate.unlock();
        updateCan = true;
    }

    if(can == nullptr){
        return;
    }

    if(can->isChanged(counter)){
        updateCan = true;
    }

    if(updateCan){
        updateCan = false;
        can->lockRead();
        print.str("");
        print<<(*can);

        msg[n] = print.str();
        n = (n+1) % nmsg;

        can->unlockRead();
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