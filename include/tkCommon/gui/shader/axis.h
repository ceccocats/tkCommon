#pragma once
/**
 * @file    axis.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw axis
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z X Y Z...]
 * 
 */
class axis : public tk::gui::shader::generic
{
    public:
        axis(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.vert";
            std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.geom";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.frag";
            
            shader.init(vertex, fragment, geometry);
        }

        ~axis(){
            
        }

        void draw(){
            glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 

            shader.use();
            shader.setMat4("modelview", modelview);
            shader.setInt("width",glutGet(GLUT_SCREEN_WIDTH));
		    shader.setInt("height",glutGet(GLUT_SCREEN_HEIGHT));

            glLineWidth(3.0);
            glDrawArrays(GL_POINTS, 0, 1);
            glLineWidth(1.0);
            
            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}