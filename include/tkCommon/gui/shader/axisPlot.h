#pragma once
/**
 * @file    axisPlot.h
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
 * @brief class that draw a pointcloud organized in 3 points [X Y Z ROLL PITCH YAW ...]
 * 
 */
class axisPlot : public tk::gui::shader::generic
{
    public:
        axisPlot(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axisPlot.vert";
            std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axisPlot.geom";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.frag";
            
            shader.init(vertex, fragment, geometry);
            
            vertexPointer.resize(4);
            vertexPointer[0] = {3,3,0};
            vertexPointer[1] = {1,1,2};
            vertexPointer[2] = {1,1,3};
            vertexPointer[3] = {1,1,4};
        }

        ~axisPlot(){
            
        }

        void draw(tk::gui::Buffer<float>* buffer, int n, float size = 1.0f){

            glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 

            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setMat4("modelview",modelview);

            glLineWidth(size);
            glDrawArrays(GL_POINTS, 0, n);
            glLineWidth(1.0);
            
            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}