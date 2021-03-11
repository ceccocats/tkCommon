#include "tkCommon/gui/drawables/Image.h"

tk::gui::Image::Image(int n, std::string name){
    this->name = name;

    this->mutex.resize(n);
    this->images.resize(n);
    this->initted.resize(n);
    this->counter.resize(n);
    this->textures.resize(n);
    this->new_ref_data.resize(n);

    for(int i = 0; i < n; i++){
        this->mutex[i] = new std::mutex();
        this->images[i] = nullptr;
        this->counter[i] = 0; 
        this->initted[i] = false;
        this->new_ref_data[i] = false;
    }   
}

tk::gui::Image::Image(std::string name, int n, ...) : Image(n,name){
    va_list arguments; 

    va_start(arguments, n);   
    for(int i = 0; i < n; i++){
        this->initted[i] = true;
        this->images[i] = va_arg(arguments, tk::data::ImageData*);
    }
    va_end(arguments);

    this->drw_has_reference = true;
}

tk::gui::Image::~Image(){

}

void 
tk::gui::Image::onInit(tk::gui::Viewer *viewer){
    for(int i = 0; i < images.size(); i++){
        if(drw_has_reference){
            this->mutex[i]->lock();
            this->textures[i] = new tk::gui::Texture<uint8_t>();
            this->textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
            this->textures[i]->setData(images[i]->data.data());
            this->mutex[i]->unlock();
        }
    }
}

void 
tk::gui::Image::updateRef(int index, tk::data::ImageData* img){
    this->mutex[index]->lock();
    this->images[index] = img;
    this->new_ref_data[index] = true;
    this->mutex[index]->unlock();
}

void 
tk::gui::Image::updateRef(tk::data::ImageData* img){
    updateRef(img->header.sensorID,img);
}

void 
tk::gui::Image::dataRef(tk::gui::Viewer *viewer){
    for(int i = 0; i < images.size(); i++){
        if(this->new_ref_data[i]){
            this->mutex[i]->lock();
            if(!this->initted[i]){
                this->initted[i]  = true;
                this->textures[i] = new tk::gui::Texture<uint8_t>();
                this->textures[i]->init(images[i]->width, images[i]->height, images[i]->channels);
            }
            this->textures[i]->setData(images[i]->data.data());
            this->new_ref_data[i] = false;
            this->mutex[i]->unlock();
        }
    }
}

void 
tk::gui::Image::dataUpd(tk::gui::Viewer *viewer){
    for(int i = 0; i < images.size(); i++){
        if(images[i]->isChanged(this->counter[i])){
            this->images[i]->lockRead();
            this->textures[i]->setData(images[i]->data.data());
            this->images[i]->unlockRead();
        }
    }
}

void 
tk::gui::Image::draw(tk::gui::Viewer *viewer){
    if(drw_has_reference){
        dataRef(viewer);
    }else{
        dataUpd(viewer);
    }

    //For each img
    for(int i = 0; i< images.size(); i++){
        ImGui::Begin(name.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
        if(this->initted[i]){
            float imgX = ImGui::GetWindowSize().x-20;
            float imgY = imgX / ((float)textures[i]->width / textures[i]->height);
            ImGui::Image((void*)(intptr_t)textures[i]->id(), ImVec2(imgX, imgY));
            ImGui::Separator();
        }
        ImGui::End();
    }
}

bool 
tk::gui::Image::isAsyncedCopied(int idx){
    this->mutex[idx]->lock();
    bool s = this->new_ref_data[idx];
    this->mutex[idx]->unlock();
    return s;
}

void 
tk::gui::Image::imGuiInfos(){
    for(int i = 0; i< images.size(); i++){
        std::stringstream print;
        if(initted[i]){
            this->mutex[i]->lock();
            print<<(*images[i]);
            this->mutex[i]->unlock();
            ImGui::Text("%s\n\n",print.str().c_str());
        }
        print.clear();
    }
}

void 
tk::gui::Image::onClose(){
    for(int i = 0; i< images.size(); i++){
        textures[i]->release();
        delete mutex[i];
    }
}