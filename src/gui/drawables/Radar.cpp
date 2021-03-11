#include "tkCommon/gui/drawables/Radar.h"


tk::gui::Radar::Radar(const std::string& name){
    this->name  = name;
    far_drw     = new tk::gui::Cloud4f("far");
    near_drw    = new tk::gui::Cloud4f("near");

    init(1);
}

tk::gui::Radar::Radar(tk::data::RadarData* radar, const std::string& name) : Radar(name){
    this->data[0] = radar;
}

tk::gui::Radar::~Radar(){

}

void 
tk::gui::Radar::onInit(tk::gui::Viewer *viewer){
    far_drw->pointSize  = 5.0f;
    near_drw->pointSize = 5.0f;
    
    viewer->add(far_drw);
    viewer->add(near_drw);
}

void 
tk::gui::Radar::updateData(int i,tk::gui::Viewer *viewer){
    tk::data::RadarData* radar = (tk::data::RadarData*)data[0];
    far_drw->updateRef(&radar->far);
    far_drw->updateRef(&radar->near);
    print.str("");
    print<<(*radar);
}

void 
tk::gui::Radar::imGuiSettings(){
    ImGui::BeginGroup();
    ImGui::BeginChild("FAR", ImVec2{-1, ImGui::GetContentRegionAvail().y/2}, true);
    ImGui::TextDisabled("FAR");
    far_drw->imGuiSettings();
    ImGui::EndChild();
    ImGui::BeginChild("NEAR", ImVec2{-1, ImGui::GetContentRegionAvail().y}, true);
    ImGui::TextDisabled("NEAR");
    near_drw->imGuiSettings();
    ImGui::EndChild();
    ImGui::EndGroup(); 
}

void 
tk::gui::Radar::imGuiInfos(){
    ImGui::Text("%s",print.str().c_str());
}
