#include "tkCommon/gui/drawables/Image.h"

tk::gui::Image::Image(int n, std::string name){
    this->textures.resize(n);
    this->images.resize(n);
    this->updates.resize(n);
    this->ready.resize(n);
    this->counter.resize(n);
    for(int i = 0; i < n; i++){
        this->updates[i] = false;
        this->ready[i]   = false;
        this->counter[i] = 0; 
    }   
    this->name = name;
}

tk::gui::Image::Image(std::string name, int n, ...){
    va_list arguments; 

    va_start(arguments, n);   
    this->textures.resize(n);
    this->images.resize(n);
    this->updates.resize(n);
    this->ready.resize(n);
    this->counter.resize(n);

    for(int i = 0; i < n; i++){
        this->images[i]  = va_arg(arguments, tk::data::ImageData*);
        this->updates[i] = false;
        this->ready[i]   = true;
        this->counter[i] = 0;
    }
    va_end(arguments);
    this->name = name;
}

tk::gui::Image::~Image(){

}

void 
tk::gui::Image::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < images.size(); i++){
        if(this->ready[i] == true){
            textures[i] = new tk::gui::Texture<uint8_t>();
            textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
            textures[i]->setData(images[i]->data.data());
        }

    }
}

void 
tk::gui::Image::updateRef(int index, tk::data::ImageData* img){
    tkASSERT(index <= images.size());
    
    this->images[index]     = img;
    this->updates[index]    = true;
}

void 
tk::gui::Image::updateRef(tk::data::VectorData<tk::data::ImageData>* vecImg) {
    for (int i = 0; i < vecImg->size(); i++)
        updateRef(i, &vecImg->data[i]);
}

void 
tk::gui::Image::draw(tk::gui::Viewer *viewer){

    for(int i = 0; i< images.size(); i++){

        //init
        if(!ready[i] && updates[i]){
            this->ready[i]  = true;
            textures[i]     = new tk::gui::Texture<uint8_t>();
            textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
        }

        if(ready[i]){
            if(images[i]->isChanged(counter[i]) || updates[i]){
                //Copy data
                images[i]->lockRead();
                textures[i]->setData(images[i]->data.data());
                images[i]->unlockRead();
                this->updates[i]  = false;
            }
        }

        ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
        if(this->ready[i] == true){
            
            float imgX = ImGui::GetWindowSize().x-20;
            //int imgY = ImGui::GetWindowSize().y-35;
            //float imgX = textures[i]->width;
            float imgY = imgX / ((float)textures[i]->width / textures[i]->height);
            //ImGui::Text("%s",images[i]->header.name.c_str());
            ImGui::Image((void*)(intptr_t)textures[i]->id(), ImVec2(imgX, imgY));
            ImGui::Separator();
        }
        ImGui::End();

    }
}

void 
tk::gui::Image::imGuiInfos(){
    for(int i = 0; i< images.size(); i++){
        std::stringstream print;
        if(ready[i] == true){
            print<<(*images[i]);
            //ImGui::Text("%s",images[i]->header.name.c_str());
            ImGui::Text("%s\n\n",print.str().c_str());
        }
        print.clear();
    }
}

void 
tk::gui::Image::onClose(){
    for(int i = 0; i< images.size(); i++)
        textures[i]->release();
}

std::string 
tk::gui::Image::toString(){
    return name;
}