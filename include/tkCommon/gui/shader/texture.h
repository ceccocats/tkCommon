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
    public:
        bool init(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false) return false;

            vertexPointer.push_back({3,5,0});
		    vertexPointer.push_back({2,5,3});
            return true;
        }

        void draw(tk::gui::Texture<uint8_t>* text, tk::gui::Buffer<float>* buffer, int triangles){

            glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview));

            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setMat4("modelview",modelview);
            text->use();
            buffer->use();

            glDrawElements(GL_TRIANGLES, triangles, GL_UNSIGNED_INT, 0);

            buffer->unuse();
            text->unuse();
            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}