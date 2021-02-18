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
        texture(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            //if(status == false) return false;

            vertexPointer.push_back({3,5,0});
		    vertexPointer.push_back({2,5,3});
            //return true;
        }

        void draw(glm::mat4& modelview, tk::gui::Texture<uint8_t>* text, tk::gui::Buffer<float>* buffer, int triangles){

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

        bool close(){
            return shader.close();
        }
};

}}}