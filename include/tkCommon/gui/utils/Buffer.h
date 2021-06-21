#pragma once
/**
 * @file    Buffer.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that handle GL_ARRAY_BUFFER for opengl shaders 
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */
#include <GL/glew.h>

#include "tkCommon/exceptions.h"

namespace tk { namespace gui {

struct vertexAttribs_t{
    int vertexSize;
    int dataSize;
    int offset;
};

template <class T>
class Buffer
{
    private:
        unsigned int VAO;
        unsigned int VBO;
        unsigned int EBO;

        int _size    = 0;
        int lastSize = 0;

        bool initEBO    = false;
        int sizeEBO     = 0;

        bool intted     = false;

        GLenum          type;
        GLenum          memType;

    public:

        /**
         * Init method
         * 
         * @param bool static draw
         * 
         */
        void init(bool staticDraw = false);


        /**
         * Method that return if EBO is setted
         * 
         * @return  if EBO is setted
         */
        bool hasEBO() const;

        /**
         * Method for set data in GL_ARRAY_BUFFER
         * 
         * @param T*    pointer to data
         * 
         * @param int   lenght
         */
        void setData(T* data, int lenght, int offset = 0);

        /**
         * Method for set indices in GL_ELEMENT_ARRAY_BUFFER
         * 
         * @param unsigned int* pointer to data
         * 
         * @param int           lenght
         */
        void setIndexVector(unsigned int* data, int lenght);

        /**
         * Method for set vertex attributes pointer array
         * 
         * @param std::vector<vertexAttribs_t>&  vector contains vertex attributes pointer
         */
        void setVertexAttribs(std::vector<vertexAttribs_t>& vertexAttribs) const;

        /**
         * use method for get size of last setData
         */
        int size() const;

        /**
         * empty method
         */
        bool empty() const;

        /**
         * use method for set GL_ARRAY_BUFFER in shader
         */
        void use() const;

        /**
         * unuse method for unset GL_ARRAY_BUFFER in shader
         */
        void unuse() const;

        /**
         * release data method
         */
        void release();
};


template <typename T>
void Buffer<T>::use() const {
    glBindVertexArray(VAO);
}

template <typename T>
void Buffer<T>::unuse() const {
    glBindVertexArray(0);
}

template <typename T>
int Buffer<T>::size() const {
    return lastSize;
}

template <typename T>
void Buffer<T>::release(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
}

template <typename T>
bool Buffer<T>::hasEBO() const {
    return initEBO;
}

template <typename T>
bool Buffer<T>::empty() const {
    return _size == 0;
}

template <typename T>
void Buffer<T>::setIndexVector(unsigned int* data, int lenght){

    //init
    if(initEBO == false){
        initEBO = true;
        sizeEBO = lenght;

        glBindVertexArray(VAO);
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, lenght * sizeof(unsigned int), data, memType); 
        glBindVertexArray(0);  
        return;    
    }

    glBindVertexArray(VAO);

    //check dim
    if(sizeEBO < lenght){
        sizeEBO = lenght;

        //delete and realloc
        glDeleteBuffers(1, &EBO);
        glGenBuffers(1, &EBO);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeEBO  * sizeof(unsigned int), data, memType);

    }else{

        //copy sub data
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, lenght * sizeof(unsigned int), data);
    }

    glBindVertexArray(0);
}

template <typename T>
void Buffer<T>::setVertexAttribs(std::vector<vertexAttribs_t>& vertexAttribs) const {
    
    glBindVertexArray(VAO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    
    //check if vertexAttribs is set
    if(vertexAttribs.size() == 0){
        tkERR("You need to fill vertexAttribsPointer\n");
        return;
    }

    for(int i = 0; i < vertexAttribs.size(); i++) {
        glEnableVertexAttribArray(i);
        glVertexAttribPointer(i, vertexAttribs[i].vertexSize, 
            type, GL_FALSE, vertexAttribs[i].dataSize * sizeof(T), (void*)(vertexAttribs[i].offset * sizeof(T)));
    }
    glBindVertexArray(0);
}

template <typename T>
void Buffer<T>::setData(T* data, int lenght, int offset){

    lastSize = lenght + offset;

    if(intted == false){
        intted = true;

        _size = lenght + offset;

        int realOffset = offset == 0 ? 1 : offset;

        glGenVertexArrays(1, &VAO);
        glGenBuffers(1, &VBO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, _size * realOffset * sizeof(T), NULL, memType);
        glBufferSubData(GL_ARRAY_BUFFER, offset * sizeof(T), lenght * sizeof(T), data);
        glBindVertexArray(0);

        return;
    }

    glBindVertexArray(VAO);

    if(lenght+offset > _size){

        _size = lenght + offset;

        //resize buffer or copy and resize
        if(offset == 0){

            //reset buffer
            glDeleteBuffers(1, &VBO);
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, _size  * sizeof(T), data, memType);

        }else{

            //creating temp buffer to copy data
            unsigned int VBOtemp;
            glGenBuffers(1, &VBOtemp);
            glBindBuffer(GL_ARRAY_BUFFER, VBOtemp);
            glBufferData(GL_ARRAY_BUFFER, offset  * sizeof(T), NULL, memType);

            //copy data
            glBindBuffer(GL_COPY_READ_BUFFER, VBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, VBOtemp);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, offset * sizeof(T));

            //reset buffer
            glDeleteBuffers(1, &VBO);
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, _size  * sizeof(T), NULL, memType);

            //recopy data
            glBindBuffer(GL_COPY_READ_BUFFER, VBOtemp);
            glBindBuffer(GL_COPY_WRITE_BUFFER, VBO);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, offset * sizeof(T));

            //fill
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferSubData(GL_ARRAY_BUFFER, offset * sizeof(T), lenght * sizeof(T), data);

            //delete temp
            glDeleteBuffers(1, &VBOtemp);
        }

    } else{

        // copy data
        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferSubData(GL_ARRAY_BUFFER, offset * sizeof(T), lenght * sizeof(T), data);
    }

    glBindVertexArray(0);
}

}}