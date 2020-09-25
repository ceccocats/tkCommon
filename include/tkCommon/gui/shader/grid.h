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

#include "tkCommon/gui/Shader.h"
#include "tkCommon/gui/Color.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z X Y Z...]
 * 
 */
class grid
{
    private:
        tk::gui::Shader           shader;

        glm::mat4 modelview;
        int   n = 50;
        float dim = 1.0;

    public:
        bool init(){
            std::string vertex      = std::string(TKPROJ_PATH) + "include/tkCommon/gui/shader/glsl/grid.vert";
            std::string fragment    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/shader/glsl/grid.frag";        
            return shader.init(vertex, fragment);
        }

        void draw(){
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
        }

        bool close(){
            return shader.close();
        }
};

}}}