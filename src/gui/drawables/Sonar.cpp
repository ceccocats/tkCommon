#include "tkCommon/gui/drawables/Sonar.h"

tk::gui::Sonar::Sonar(std::string name, tk::data::SensorData* sonar){
    this->data = sonar;
    this->name = name;
}

tk::gui::Sonar::~Sonar(){

}

void 
tk::gui::Sonar::onInit(tk::gui::Viewer *viewer){
    if(data != nullptr){
        data->lockRead();
        texture = new tk::gui::Texture<float>();
        tk::data::SonarData* sonar = dynamic_cast<tk::data::SonarData*>(data);
        if(sonar->image.width != 0 && sonar->image.height != 0 && sonar->image.channels != 0){
            texture->init(sonar->image.width, sonar->image.height, sonar->image.channels);
            texture->setData(sonar->image.data.data());
        }
        data->unlockRead();
    }
}

void 
tk::gui::Sonar::updateData(tk::gui::Viewer *viewer){
    if(texture == nullptr){
        onInit(viewer);
    }
    this->tf = data->header.tf;
    tk::data::SonarData* sonar = dynamic_cast<tk::data::SonarData*>(data);
    if(sonar->image.width != texture->width || sonar->image.height != texture->height || sonar->image.channels != texture->channels){
        texture->release();
        delete texture;
        texture = new tk::gui::Texture<float>();
        tk::data::SonarData* sonar = dynamic_cast<tk::data::SonarData*>(data);
        texture->init(sonar->image.width, sonar->image.height, sonar->image.channels);
    }
    texture->setData(sonar->image.data.data());
}

void 
tk::gui::Sonar::drawData(tk::gui::Viewer *viewer){

    ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
    if(texture != nullptr){
        float imgX = ImGui::GetWindowSize().x-20;
        print.str("");

        float imgY = imgX / ((float)texture->width / texture->height);
        ImGui::Image((void*)(intptr_t)texture->id(), ImVec2(imgX, imgY));

        tk::data::SonarData* sonar = dynamic_cast<tk::data::SonarData*>(data);
        print<<(*sonar);

        ImGui::Separator();
    }
    ImGui::End();
}

void 
tk::gui::Sonar::imGuiInfos(){
    ImGui::Text("%s\n\n",print.str().c_str());
}

void 
tk::gui::Sonar::onClose(){
    if(texture != nullptr){
        texture->release();
        delete texture;
    }
}