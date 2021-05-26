#include "tkCommon/gui/drawables/ImageTexture.h"

tk::gui::ImageTexture::ImageTexture(std::string name,tk::data::SensorData* img){
    this->data = img;
    this->name = name;
    shader = tk::gui::shader::texture::getInstance();
}

tk::gui::ImageTexture::~ImageTexture(){

}

void 
tk::gui::ImageTexture::onInit(tk::gui::Viewer *viewer){
    if(data != nullptr){
        data->lockRead();
        if(data->header.type == tk::data::DataType::IMAGEU8){
            textureType = 0;
            texture = new tk::gui::Texture<uint8_t>();
            tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
            tk::data::ImageDataU8* imgU8 = (tk::data::ImageDataU8*)data;
            textU8->init(imgU8->width, imgU8->height, imgU8->channels);
            textU8->setData(imgU8->data.data());
        }
        if(data->header.type == tk::data::DataType::IMAGEU16){
            textureType = 1;
            texture = new tk::gui::Texture<uint16_t>();
            tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
            tk::data::ImageDataU16* imgU16 = (tk::data::ImageDataU16*)data;
            textU16->init(imgU16->width, imgU16->height, imgU16->channels);
            textU16->setData(imgU16->data.data());
        }
        if(data->header.type == tk::data::DataType::IMAGEF){
            textureType = 2;
            texture = new tk::gui::Texture<float>();
            tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
            tk::data::ImageDataF* imgF = (tk::data::ImageDataF*)data;
            textF->init(imgF->width, imgF->height, imgF->channels);
            textF->setData(imgF->data.data());
        }
        data->unlockRead();
        if(texture == nullptr){
            tkERR("Image type not supported\n");
            tkDBG((int)data->header.type<<"\n");
        }
    }
}

void 
tk::gui::ImageTexture::setPos(std::vector<tk::common::Vector3<float>> fourVertices){
    if(fourVertices.size() != 4){
        tkERR("You must pass 4 vertices\n");
        return;
    }
    if(pos == nullptr){
        pos = new tk::gui::Buffer<float>();
        pos->init(true);
        indices[0] = 0;
        indices[1] = 1;
        indices[2] = 2;
        indices[3] = 0;
        indices[4] = 3;
        indices[5] = 2;
    }  

    vertices[0] =  fourVertices[0].x();
    vertices[1] =  fourVertices[0].y();
    vertices[2] =  fourVertices[0].z();
    vertices[3] =  0.0f;
    vertices[4] =  1.0f;
    vertices[5] =  fourVertices[1].x();
    vertices[6] =  fourVertices[1].y();
    vertices[7] =  fourVertices[1].z();
    vertices[8] =  1.0f;
    vertices[9] =  1.0f;
    vertices[10] = fourVertices[2].x();
    vertices[11] = fourVertices[2].y();
    vertices[12] = fourVertices[2].z();
    vertices[13] = 1.0f;
    vertices[14] = 0.0f;
    vertices[15] = fourVertices[3].x();
    vertices[16] = fourVertices[3].y();
    vertices[17] = fourVertices[3].z();
    vertices[18] = 0.0f;
    vertices[19] = 0.0f;

    pos->setData(vertices,20);
    pos->setIndexVector(indices,6);
}

void 
tk::gui::ImageTexture::updateData(tk::gui::Viewer *viewer){
    if(texture == nullptr){
        onInit(viewer);
    }
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
        textF->setData(imgF->data.cpu.data);
    }
}

void 
tk::gui::ImageTexture::drawData(tk::gui::Viewer *viewer){

    print.str("");
    if(pos != nullptr && textureType == 0){
        tk::gui::Texture<uint8_t>* textU8 = (tk::gui::Texture<uint8_t>*)texture;
        tk::gui::shader::texture* shaderTexture = (tk::gui::shader::texture*)shader;
        shaderTexture->draw<uint8_t>(drwModelView,textU8,pos,6);
        tk::data::ImageDataU8* imgU8 = (tk::data::ImageDataU8*)data;
        print<<(*imgU8);
    }
    if(textureType == 1){
        tk::gui::Texture<uint16_t>* textU16 = (tk::gui::Texture<uint16_t>*)texture;
        tk::gui::shader::texture* shaderTexture = (tk::gui::shader::texture*)shader;
        shaderTexture->draw<uint16_t>(drwModelView,textU16,pos,6);
        tk::data::ImageDataU16* imgU16 = (tk::data::ImageDataU16*)data;
        print<<(*imgU16);
    }
    if(textureType == 2){
        tk::gui::Texture<float>* textF = (tk::gui::Texture<float>*)texture;
        tk::gui::shader::texture* shaderTexture = (tk::gui::shader::texture*)shader;
        shaderTexture->draw<float>(drwModelView,textF,pos,6);
        tk::data::ImageDataF* imgF = (tk::data::ImageDataF*)data;
        print<<(*imgF);
    }
        
}

void 
tk::gui::ImageTexture::imGuiInfos(){
    ImGui::Text("%s\n\n",print.str().c_str());
}

void 
tk::gui::ImageTexture::onClose(){
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
    if(pos != nullptr){
        pos->release();
        delete  pos;
    }
}