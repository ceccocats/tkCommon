#pragma once
/**
 * @file    texture.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw a texture with shader
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a texture, data organized in [X Y Z textX textY ...]
 * 
 */
class texture : public tk::gui::shader::generic
{
    private:
        static int users;
        texture();

    public:
        static texture* getInstance(){
            static texture instance;
            users++;
            return &instance;
        }
        template <class T>
        void draw(glm::mat4& modelview, tk::gui::Texture<T>* text, tk::gui::Buffer<float>* buffer, int triangles);
        void close();
};


template <typename T> 
void texture::draw(glm::mat4& modelview, tk::gui::Texture<T>* text, tk::gui::Buffer<float>* buffer, int triangles){

    buffer->setVertexAttribs(vertexPointer);

    shader.use();
    shader.setMat4("modelview",modelview);
    text->use();
    buffer->use();

    if(buffer->hasEBO()){
        glDrawElements(GL_TRIANGLES, triangles, GL_UNSIGNED_INT, 0);
    }else{
        glDrawArrays(GL_TRIANGLES, 0, triangles);
    }

    buffer->unuse();
    text->unuse();
    shader.unuse();

    glCheckError();
}

}}}