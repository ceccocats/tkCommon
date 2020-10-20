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

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a lines formed by [X Y Z R G B A ...]
 * 
 */
class linesMonocolor : public tk::gui::shader::generic
{
    private:
        glm::vec4 linesColor;

    public:

        struct line_color_t{
            float x;
            float y;
            float z;

            line_color_t& operator =(const line_color_t& a){
                x = a.x;
                y = a.y;
                z = a.z;
                return *this;
            }
        };

    public:
        
        linesMonocolor(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/linesMonocolor.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/linesMonocolor.frag";
            
            shader.init(vertex, fragment, geometry);

            vertexPointer.resize(1);
            vertexPointer[0] = {3,3,0};
        }

        void draw(tk::gui::Buffer<float>* buffer, int n, int size = 1.0f, 
                tk::gui::Color_t color = tk::gui::color::WHITE, GLenum linemode = GL_LINE_STRIP){

		    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview));

            buffer->setVertexAttribs(vertexPointer);

            std::memcpy(glm::value_ptr(linesColor), color.color, sizeof(linesColor));

            shader.use();
            shader.setMat4("modelview",modelview);
            shader.setVec4("color",linesColor);

            buffer->use();
            glLineWidth(size);
            glDrawArrays(linemode, 0, n);
            glLineWidth(1.0f);
            buffer->unuse();

            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}