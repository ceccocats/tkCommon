#include "tkCommon/gui/drawables/Image.h"

tk::gui::Image::Image(std::string name, std::string imguiName, tk::data::SensorData* img){
    if(imguiName.empty()){
        this->imguiName = name;
    }else{
        this->imguiName = imguiName;
    }
    this->data = img;
    this->name = name;
}

tk::gui::Image::~Image(){

}

void 
tk::gui::Image::onInit(tk::gui::Viewer *viewer){
    if(data != nullptr){
        data->lockRead();
        if(data->header.type == tk::data::DataType::IMAGEU8){
            textureType = 0;
            texture = new tk::gui::Texture<uint8_t>();
            tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
            tk::data::ImageDataU8* imgU8 = (tk::data::ImageDataU8*)data;

            if (imgU8->width != 0 && imgU8->height != 0 && imgU8->channels != 0) {
                textU8->init(imgU8->width, imgU8->height, imgU8->channels);
                textU8->setData(imgU8->data.data());
            }
        }
        if(data->header.type == tk::data::DataType::IMAGEU16){
            textureType = 1;
            texture = new tk::gui::Texture<uint16_t>();
            tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
            tk::data::ImageDataU16* imgU16 = (tk::data::ImageDataU16*)data;

            if (imgU16->width != 0 && imgU16->height != 0 && imgU16->channels != 0) {
                textU16->init(imgU16->width, imgU16->height, imgU16->channels);
                textU16->setData(imgU16->data.data());
            }
        }
        if(data->header.type == tk::data::DataType::IMAGEF){
            textureType = 2;
            texture = new tk::gui::Texture<float>();
            tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
            tk::data::ImageDataF* imgF = (tk::data::ImageDataF*)data;

            if (imgF->width != 0 && imgF->height != 0 && imgF->channels != 0) {
                textF->init(imgF->width, imgF->height, imgF->channels);
                textF->setData(imgF->data.data());
            }
        }
        data->unlockRead();
        if(texture == nullptr){
            tkERR("Image type not supported\n");
            tkDBG((int)data->header.type<<"\n");
        }
    }
}

void 
tk::gui::Image::updateData(tk::gui::Viewer *viewer){
    if(texture == nullptr){
        onInit(viewer);
    }
    this->tf = data->header.tf;
    if(textureType == 0){
        tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
        tk::data::ImageDataU8* imgU8 = (tk::data::ImageDataU8*)data;
        if(imgU8->width != textU8->width || imgU8->height != textU8->height || imgU8->channels != textU8->channels){
            textU8->release();
            delete texture;
            texture = new tk::gui::Texture<uint8_t>();
            textU8 = (tk::gui::Texture<uint8_t>*)texture;
            textU8->init(imgU8->width, imgU8->height, imgU8->channels);
        }
        if (imgU8->width != 0 && imgU8->height != 0 && imgU8->channels != 0)
            textU8->setData(imgU8->data.cpu.data);
    }
    if(textureType == 1){
        tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
        tk::data::ImageDataU16* imgU16 = (tk::data::ImageDataU16*)data;
        if(imgU16->width != textU16->width || imgU16->height != textU16->height || imgU16->channels != textU16->channels){
            textU16->release();
            delete texture;
            texture = new tk::gui::Texture<uint16_t>();
            textU16 = (tk::gui::Texture<uint16_t>*)texture;
            textU16->init(imgU16->width, imgU16->height, imgU16->channels);
        }
        if (imgU16->width != 0 && imgU16->height != 0 && imgU16->channels != 0) 
            textU16->setData(imgU16->data.cpu.data);
    }
    if(textureType == 2){
        tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
        tk::data::ImageDataF* imgF = (tk::data::ImageDataF*)data;
        if(imgF->width != textF->width || imgF->height != textF->height || imgF->channels != textF->channels){
            textF->release();
            delete texture;
            texture = new tk::gui::Texture<float>();
            textF = (tk::gui::Texture<float>*)texture;
            textF->init(imgF->width, imgF->height, imgF->channels);
        }
        if (imgF->width != 0 && imgF->height != 0 && imgF->channels != 0)
            textF->setData(imgF->data.cpu.data);
    }
}

void 
tk::gui::Image::drawData(tk::gui::Viewer *viewer){

    ImGui::Begin(imguiName.c_str(), NULL, ImGuiWindowFlags_NoScrollbar);
    if(texture != nullptr){
        float imgX = ImGui::GetWindowSize().x-20;
        
        print.str("");
        if(textureType == 0){
            tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
            float imgY = imgX / ((float)textU8->width / textU8->height);
            ImGui::Image((void*)(intptr_t)textU8->id(), ImVec2(imgX, imgY));

            tk::data::ImageDataU8* imgU8 = (tk::data::ImageDataU8*)data;
            print<<(*imgU8);
        }
        if(textureType == 1){
            tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
            float imgY = imgX / ((float)textU16->width / textU16->height);
            ImGui::Image((void*)(intptr_t)textU16->id(), ImVec2(imgX, imgY));

            tk::data::ImageDataU16* imgU16 = (tk::data::ImageDataU16*)data;
            print<<(*imgU16);
        }
        if(textureType == 2){
            tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
            float imgY = imgX / ((float)textF->width / textF->height);
            ImGui::Image((void*)(intptr_t)textF->id(), ImVec2(imgX, imgY));

            tk::data::ImageDataF* imgF = (tk::data::ImageDataF*)data;
            print<<(*imgF);
        }
        ImGui::Separator();
    }
    ImGui::End();
}

void 
tk::gui::Image::imGuiInfos(){
    ImGui::Text("%s\n\n",print.str().c_str());
}

void 
tk::gui::Image::onClose(){
    if(textureType == 0){
        tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
        textU8->release();
    }
    if(textureType == 1){
        tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
        textU16->release();
    }
    if(textureType == 2){
        tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
        textF->release();
    }
    if(texture != nullptr){
        delete texture;
    }
}