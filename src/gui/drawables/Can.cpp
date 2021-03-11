#include "tkCommon/gui/drawables/Can.h"

tk::gui::Can::Can(std::string name){
    this->name              = name;
    msg.setDim(n_msg);
}

tk::gui::Can::Can(tk::data::CanData* data, std::string name){
    this->data              = data;
    this->name              = name;
    msg.setDim(n_msg);
}

tk::gui::Can::~Can(){
}


void
tk::gui::Can::updateData(tk::gui::Viewer *viewer){
    print.str("");
    print<<(*data);
    msg.add(print.str());
}

void 
tk::gui::Can::imGuiInfos(){
    for(int i = 0; i < msg.size(); i++){
        ImGui::Text("%s",msg.array[i].c_str());
    }
}

void 
tk::gui::Can::imGuiSettings(){
    if(ImGui::SliderInt("Last messages",&n_msg,1,MAX_MSG)){
        msg.setDim(n_msg);
    }
}

void 
tk::gui::Can::onClose(){
    msg.clear();
}