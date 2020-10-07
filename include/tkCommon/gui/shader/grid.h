#pragma once
/**
 * @file    grid.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw grid
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw grid
 * 
 */
class grid : public tk::gui::shader::generic
{
    public:
        grid(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/grid.vert";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/grid.frag";        
            shader.init(vertex, fragment);
        }
        
        ~grid(){

        }

        void draw(float dim = 1.0f, int n = 50){
            glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 

            shader.use();
            shader.setMat4("modelview", modelview);
            shader.setInt("n", n);
            shader.setFloat("dim", dim);

            shader.setInt("dir", 0);
            glDrawArrays(GL_LINES, 0, n*2 +2);
            shader.setInt("dir", 1);
            glDrawArrays(GL_LINES, 0, n*2 +2);

            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}