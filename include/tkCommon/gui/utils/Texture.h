#pragma once
/**
 * @file    Texture.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that handle GL_TEXTURE_2D for opengl code 
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <GL/glew.h>

#include "tkCommon/exceptions.h"

namespace tk { namespace gui {

class TextureGeneric{
};

template <class T>
class Texture : public TextureGeneric
{
    private:
        GLenum          format;
        GLenum          type;

        bool            initRendering = false;
        GLuint          framebuffer;
        GLuint          RBO;
        GLint           Viewport[4];

        bool            anti_aliasing;
        GLuint          textureColorBufferMultiSampled;
        GLuint          intermediateFBO;

    public:
        GLuint          texture;
        int             width;
        int             height;

        /**
         * Init method
         * 
         * @param int   image width
         * @param int   image geight
         * @param int   channels
         */
        void init(int width, int height, int channels, bool anti_aliasing = false);

        /**
         * Method for set texture data
         * 
         * @param T*    pointer to data
         */
        void setData(T* data);

         /**
         * Method for get texture id
         * 
         * @return      texture id
         */
        unsigned int id();

        /**
         * use method for set GL_TEXTURE_2D in shader
         */
        void use();

        /**
         * unuse method for unset GL_TEXTURE_2D in shader
         */
        void unuse();

        /**
         *  Method for set texture as rendering frame
         */
        void useForRendering();

        /**
         * Method for unset texture as rendering frame
         */
        void unuseRendering();

        /**
         * release data method
         */
        void release();

    private:

        /**
         * Internal method for generateTexture
         * 
         * @param GLenum    texture format
         */
        void generateTexture(GLenum format);

        /**
         * Internal method for rendering on texture without antialiasing
         */
        void renderWithoutAntialiasing();

        /**
         * Internal method for rendering on texture with antialiasing
         */
        void renderWithAntialiasing();
};

template <typename T>
void Texture<T>::use(){
    glBindTexture(GL_TEXTURE_2D, texture);
}

template <typename T>
void Texture<T>::unuse(){
    glBindTexture(GL_TEXTURE_2D, 0);
}

template <typename T>
void Texture<T>::useForRendering(){

    if(anti_aliasing == true){
        this->renderWithAntialiasing();
    }else{
        this->renderWithoutAntialiasing();
    }
}

template <typename T>
void Texture<T>::unuseRendering(){

    if(anti_aliasing == true){
        glBindFramebuffer(GL_READ_FRAMEBUFFER, framebuffer);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, intermediateFBO);
        glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT,  GL_LINEAR);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(Viewport[0], Viewport[1], (GLsizei)Viewport[2], (GLsizei)Viewport[3]);
}

template <typename T>
void Texture<T>::release(){
    glDeleteTextures(1,&texture);
}

template <typename T>
void Texture<T>::setData(T* data){
    use();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, data);
    unuse();
}

template <typename T>
GLuint Texture<T>::id(){
    return texture;
}

template <typename T>
void Texture<T>::generateTexture(GLenum format){

    glGenTextures(1, &texture);
    
    use();
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    glTexStorage2D(GL_TEXTURE_2D, 1, format, width, height);
    unuse();
}

template <typename T>
void Texture<T>::renderWithoutAntialiasing(){
    if(initRendering == false){
        initRendering = true;

        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        glGenRenderbuffers(1, &RBO);
        glBindRenderbuffer(GL_RENDERBUFFER, RBO);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO); 
        
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
            tkERR("Error in rendering on texture\n");
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glGetIntegerv(GL_VIEWPORT, Viewport);
    glViewport(0,0,width,height);
}


template <typename T>
void Texture<T>::renderWithAntialiasing(){
    if(initRendering == false){
        initRendering = true;

        glGenFramebuffers(1, &framebuffer);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

        // create a multisampled color attachment texture
        glGenTextures(1, &textureColorBufferMultiSampled);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled);
        glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 6, format, width, height, GL_TRUE);
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, textureColorBufferMultiSampled, 0);


        // create a (also multisampled) renderbuffer object for depth and stencil attachments
        glGenRenderbuffers(1, &RBO);
        glBindRenderbuffer(GL_RENDERBUFFER, RBO);
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, 6, GL_DEPTH24_STENCIL8, width, height);
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, RBO);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
            tkERR("Error in rendering on texture\n");
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // configure second post-processing framebuffer
        glGenFramebuffers(1, &intermediateFBO);
        glBindFramebuffer(GL_FRAMEBUFFER, intermediateFBO);

        // create a color attachment texture
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){
            tkERR("Error in rendering on texture\n");
            return;
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glGetIntegerv(GL_VIEWPORT, Viewport);
    glViewport(0,0,width,height);
}    

}}