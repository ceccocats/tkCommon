#include "tkCommon/gui/drawables/ReplayInfo.h"

tk::gui::ReplayInfo::ReplayInfo(tk::sensors::LogManager* manager){
    this->manager = manager;
    this->name    = "ReplayInfo";
}

void 
tk::gui::ReplayInfo::imGuiSettings(){
    if(manager != nullptr){
        if(ImGui::SliderFloat("Replay speed",&speed,0.1,20,"%.1fx")){
            manager->speed = speed;
        }
        if (ImGui::Button("start/stop")){
            manager->startStop();
        }
    }
}

void 
tk::gui::ReplayInfo::imGuiInfos(){
    if(manager != nullptr){
        ImGui::Text("Replay clock: %llu",manager->getTick());
    }
}

void 
tk::gui::ReplayInfo::onClose(){
    manager->exitStop();
}