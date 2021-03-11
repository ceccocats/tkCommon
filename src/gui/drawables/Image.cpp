#include "tkCommon/gui/drawables/Image.h"

tk::gui::Image::Image(int n, std::string name){
    this->name = name;

    this->initted.resize(n);
    this->textures.resize(n);

    for(int i = 0; i < n; i++){
        this->initted[i] = false;
    }   
}

tk::gui::Image::Image(std::string name, int n, ...) : Image(n,name){
    va_list arguments; 

    va_start(arguments, n);   
    for(int i = 0; i < n; i++){
        this->initted[i] = true;
        this->data[i]    = va_arg(arguments, tk::data::ImageData*);
    }
    va_end(arguments);
}

tk::gui::Image::~Image(){

}

void 
tk::gui::Image::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < data.size(); i++){
        if(initted[i]){
            tk::data::ImageData* img = (tk::data::ImageData*)data[i];
            img->lockRead();
            this->textures[i] = new tk::gui::Texture<uint8_t>();
            this->textures[i]->init(img->width, img->height, img->channels);
            this->textures[i]->setData(img->data.data());
            img->unlockRead();
        }
    }
}

void 
tk::gui::Image::updateRef(int index, tk::data::ImageData* img){
    img->header.sensorID = index;
    DataDrawable::updateRef((tk::data::SensorData*)img);
}

void 
tk::gui::Image::updateData(int i, tk::gui::Viewer *viewer){
    tk::data::ImageData* img = (tk::data::ImageData*)data[i];
    if(!this->initted[i]){
        this->initted[i]  = true;
        this->textures[i] = new tk::gui::Texture<uint8_t>();
        this->textures[i]->init(img->width, img->height, img->channels);
    }
    this->textures[i]->setData(img->data.data());
}

void 
tk::gui::Image::drawData(tk::gui::Viewer *viewer){
    print.clear();

    for(int i = 0; i< data.size(); i++){
        ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
        if(this->initted[i]){
            float imgX = ImGui::GetWindowSize().x-20;
            float imgY = imgX / ((float)textures[i]->width / textures[i]->height);
            ImGui::Image((void*)(intptr_t)textures[i]->id(), ImVec2(imgX, imgY));
            ImGui::Separator();
            tk::data::ImageData* img = (tk::data::ImageData*)data[i];
            print<<(*img);
        }
        ImGui::End();
    }
}

void 
tk::gui::Image::imGuiInfos(){
    for(int i = 0; i< data.size(); i++){
        std::stringstream print;
        if(initted[i]){
            ImGui::Text("%s\n\n",print.str().c_str());
        }
    }
}

void 
tk::gui::Image::onClose(){
    for(int i = 0; i< data.size(); i++){
        textures[i]->release();
    }
}