#include "tkCommon/gui/utils/Texture.h"

namespace tk { namespace gui {

template <>
void Texture<uint8_t>::init(int width, int height, int channels, bool anti_aliasing){

    this->width         =   width;
    this->height        =  height;
    this->anti_aliasing = anti_aliasing;

    if(channels > 4 || channels < 1){
        tkERR("You must set 1, 2, 3 or 4 channel. Abort\n");
    }

    type =  GL_UNSIGNED_BYTE;

    if(channels == 1){
        format = GL_LUMINANCE;
        this->generateTexture(GL_LUMINANCE8);
    }

    if(channels == 2){
        format = GL_RG;
        this->generateTexture(GL_RG8);
    }

    if(channels == 3){
        format = GL_RGB;
        this->generateTexture(GL_RGB8);
    }

    if(channels == 4){
        format = GL_RGBA;
        this->generateTexture(GL_RGBA8);
    }
}

template <>
void Texture<float>::init(int width, int height, int channels, bool anti_aliasing){

    this->width         =   width;
    this->height        =  height;
    this->anti_aliasing = anti_aliasing;

    if(channels > 4 || channels < 1){
        tkERR("You must set 1, 2, 3 or 4 channel. Abort\n");
    }

    type =  GL_FLOAT;

    if(channels == 1){
        format = GL_LUMINANCE;
        this->generateTexture(GL_LUMINANCE8);
    }

    if(channels == 2){
        format = GL_RG;
        this->generateTexture(GL_RG16F);
    }

    if(channels == 3){
        format = GL_RGB;
        this->generateTexture(GL_RGB16F);
    }

    if(channels == 4){
        format = GL_RGBA;
        this->generateTexture(GL_RGBA16F);
    }
}

}}