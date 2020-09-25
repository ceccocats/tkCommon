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

#include "tkCommon/gui/Shader.h"
#include "tkCommon/gui/Texture.h"
#include "tkCommon/gui/Buffer.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/common.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a texture, data organized in [X Y Z textX textY ...]
 * 
 */
class texture
{
    private:
        tk::gui::Shader                       shader;
        std::vector<tk::gui::vertexAttribs_t>   vertexPointer;

        glm::mat4                               modelview;

    public:
        bool init(){
            std::string vertex      = std::string(TKPROJ_PATH) + "include/tkCommon/gui/shader/glsl/texture.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/shader/glsl/texture.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false) return false;

            vertexPointer.push_back({3,5,0});
		    vertexPointer.push_back({2,5,3});
            return true;
        }

        void draw(tk::gui::Texture<uint8_t>* texture, tk::gui::Buffer<float>* buffer, int triangles, int withModelview = false){

            if(withModelview == true){
                    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview));
            }else{
                    modelview = glm::mat4(1.0); //identity
            }

            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setMat4("modelview",modelview);
            texture->use();
            buffer->use();

            glDrawElements(GL_TRIANGLES, triangles, GL_UNSIGNED_INT, 0);

            buffer->unuse();
            texture->unuse();
            shader.unuse();
        }

        bool close(){
            return shader.close();
        }
};

}}}