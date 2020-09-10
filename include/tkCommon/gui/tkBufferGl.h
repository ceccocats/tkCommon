#pragma once
/**
 * @file    tkBufferGl.h
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
class tkBufferGl
{
    private:
        unsigned int VAO;
        unsigned int VBO;

        int size = 0;

        std::vector<vertexAttribs_t> vertexAttribs;

    public:

        /**
         * Init method
         */
        void init();

        /**
         * Method for set data in GL_ARRAY_BUFFER
         * 
         * @param T* pointer to data
         * 
         * @param int   lenght
         */
        void setData(T* data, int lenght, int offset = 0);

        /**
         * Method for set vertex attributes pointer
         * 
         * @param int   lenght of current vertex
         * 
         * @param int   offset of starting vertex position
         * 
         * @param int   lenght of total vertex in a frame
         */
        void pushVertexAttrib(int vertexSize, int offset, int dataSize);

        /**
         * Method for set vertex attributes pointer array
         * 
         * @param std::vector<vertexAttribs_t>&  vector contains vertex attributes pointer
         */
        void pushVectorVertex(std::vector<vertexAttribs_t>& vec);

        /**
         * use method for set GL_ARRAY_BUFFER in shader
         */
        void use();

        /**
         * unuse method for unset GL_ARRAY_BUFFER in shader
         */
        void unuse();

        /**
         * update vertex attributes pointer
         */
        void setVertexAttrib();

        /**
         * release data method
         */
        void release();
};

template <typename T>
void tkBufferGl<T>::init(){
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
}

template <typename T>
void tkBufferGl<T>::use(){

    glBindVertexArray(VAO);
}

template <typename T>
void tkBufferGl<T>::unuse(){
    glBindVertexArray(0);
}

template <typename T>
void tkBufferGl<T>::release(){
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
}

template <typename T>
void tkBufferGl<T>::pushVertexAttrib(int vertexSize, int offset, int dataSize){
    vertexAttribs.push_back({vertexSize, offset, dataSize});
}

template <typename T>
void tkBufferGl<T>::pushVectorVertex(std::vector<vertexAttribs_t>& vec){
    vertexAttribs = vec;
}

template <typename T>
void tkBufferGl<T>::setData(T* data, int lenght, int offset){
            
    glBindVertexArray(VAO);

    if(lenght+offset > size){

        size = lenght + offset;

        //resize buffer or copy and resize
        if(offset == 0){

            //reset buffer
            glDeleteBuffers(1, &VBO);
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, size  * sizeof(T), data, GL_DYNAMIC_DRAW);

        }else{

            //creating temp buffer to copy data
            unsigned int VBOtemp;
            glGenBuffers(1, &VBOtemp);
            glBindBuffer(GL_ARRAY_BUFFER, VBOtemp);
            glBufferData(GL_ARRAY_BUFFER, offset  * sizeof(T), NULL, GL_DYNAMIC_DRAW);

            //copy data
            glBindBuffer(GL_COPY_READ_BUFFER, VBO);
            glBindBuffer(GL_COPY_WRITE_BUFFER, VBOtemp);
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, offset * sizeof(T));

            //reset buffer
            glDeleteBuffers(1, &VBO);
            glGenBuffers(1, &VBO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, size  * sizeof(T), NULL, GL_DYNAMIC_DRAW);

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