#pragma once
/**
 * @file    tkTexture.h
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

template <class T>
class tkTexture
{
    private:
        GLuint          texture;
        int             width;
        int             height;

        GLenum          format;
        GLenum          type;

    public:

        /**
         * Init method
         * 
         * @param int   image width
         * @param int   image geight
         * @param int   channels
         */
        void init(int width, int height, int channels);

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
         * release data method
         */
        void release();
};

template <typename T>
void tkTexture<T>::use(){
    glBindTexture(GL_TEXTURE_2D, texture);
}

template <typename T>
void tkTexture<T>::unuse(){
    glBindTexture(GL_TEXTURE_2D, 0);
}

template <typename T>
void tkTexture<T>::release(){
    glDeleteTextures(1,&texture);
}

template <typename T>
void tkTexture<T>::setData(T* data){
    use();
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, format, type, data);
    unuse();
}

template <typename T>
GLuint tkTexture<T>::id(){
    return texture;
}

}}