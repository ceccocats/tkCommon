#include "tkCommon/gui/tkTexture.h"

namespace tk { namespace gui {
template <>
void tkTexture<uint8_t>::init(int width, int height, int channels){

    this->width =   width;
    this->height =  height;

    glGenTextures(1, &texture);
    use();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if(channels > 4 && channels < 1){
        clsErr("You must set 1, 2, 3 or 4 channel. Abort\n");
    }

    type =  GL_UNSIGNED_BYTE;

    if(channels == 1){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R8, width, height);
        format = GL_R;
    }

    if(channels == 2){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG8, width, height);
        format = GL_RG;
    }

    if(channels == 3){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB8, width, height);
        format = GL_RGB;
    }

    if(channels == 4){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, width, height);
        format = GL_RGBA;
    }
    unuse();
}

template <>
void tkTexture<float>::init(int width, int height, int channels){

    this->width =   width;
    this->height =  height;
    
    glGenTextures(1, &texture);
    use();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    if(channels > 4 && channels < 1){
        clsErr("You must set 1, 2, 3 or 4 channel. Abort\n");
    }

    type =  GL_FLOAT;

    if(channels == 1){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_R16F, width, height);
        format = GL_R;
    }

    if(channels == 2){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RG16F, width, height);
        format = GL_RG;
    }

    if(channels == 3){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGB16F, width, height);
        format = GL_RGB;
    }

    if(channels == 4){
        glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA16F, width, height);
        format = GL_RGBA;
    }
    unuse();
}

}}