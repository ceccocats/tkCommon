#pragma once
/**
 * @file    stripline.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw lines from points
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/Color.h"
#include "tkCommon/common.h"

namespace tk { namespace gui { namespace components {

/**
 * @brief class that draw a lines formed by [X Y Z R G B A ...]
 * 
 */
class lines
{
    private:
        std::vector<tk::gui::vertexAttribs_t>   vertexPointer;
        tk::gui::tkShader                       shader;

        glm::mat4                               modelview;

    public:
        bool init(){
            std::string vertex      = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/lines/lines.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/lines/lines.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false)
                return false;

            vertexPointer.push_back({3,7,0});
            vertexPointer.push_back({4,7,3});
            return true;
        }

        void draw(tk::gui::tkBufferGl<float>* buffer, int n, int size = 1.0f,GLenum linemode = GL_LINE_STRIP, bool withModelview = true){

		    if(withModelview == true){
                    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview));
            }else{
                    modelview = glm::mat4(1.0); //identity
            }

            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setMat4("modelview",modelview);

            buffer->use();
            glLineWidth(size);
            glDrawArrays(linemode, 0, n);
            glLineWidth(1.0f);
            buffer->unuse();

            shader.unuse();
        }

        bool close(){
            return shader.close();
        }


};

}}}