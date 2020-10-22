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

#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/Color.h"

namespace tk { namespace gui { namespace components {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z X Y Z...]
 * 
 */
class axis
{
    private:
        tk::gui::tkShader                       shader;

        glm::mat4                               modelview;

    public:
        bool init(){
            std::string vertex      = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/axis/axis.vert";
            std::string geometry    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/axis/axis.geom";
            std::string fragment    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/axis/axis.frag";
            
            return shader.init(vertex, fragment, geometry);
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
        }

        bool close(){
            return shader.close();
        }
};

}}}