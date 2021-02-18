#pragma once
/**
 * @file    cubes3D.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw cubes
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw cubes organized in 13 values
 *          [Xcenter. Ycenter, Zcenter, Xsize, Ysize, Zsize, roll, pitch, yaw, red, green, blue, alpha]
 * 
 */
class cubes3D : public tk::gui::shader::generic
{
    public:
        bool init(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.vert";
            std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.geom";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false)
                return false;

            //Xcenter. Ycenter, Zcenter
            vertexPointer.push_back({3,13,0});
            //Xsize, Ysize, Zsize
            vertexPointer.push_back({3,13,3});
            //roll, pitch, yaw
            vertexPointer.push_back({3,13,6});
            //red, green, blue, alpha
            vertexPointer.push_back({4,13,9});
            return true;
        }

        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3& lightPos){

            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setMat4("modelview",modelview);
            shader.setVec3("lightPos", lightPos);

            buffer->use();
            glDrawArrays(GL_POINTS, 0, n);
            buffer->unuse();

            shader.unuse();

            glCheckError();
        }

        bool close(){
            return shader.close();
        }


};

}}}