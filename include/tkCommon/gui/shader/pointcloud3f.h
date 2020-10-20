#pragma once
/**
 * @file    pointcloud3f.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw pointcloud formed by 3f with shaders
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
class pointcloud3f  : public tk::gui::shader::generic
{
    private:
        glm::vec4 pointColor;

    public:
        bool init(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloud3f.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudFrag/pointcloudFrag_uniformColor.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false)
                return false;

            vertexPointer.push_back({3,3,0});
            return true;
        }

        void draw(tk::gui::Buffer<float>* buffer, int n, tk::gui::Color_t color = tk::gui::color::WHITE){

		    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 
            buffer->setVertexAttribs(vertexPointer);

            std::memcpy(glm::value_ptr(pointColor), color.color, sizeof(pointColor));

            shader.use();
            shader.setMat4("modelview",modelview);
            shader.setVec4("color",pointColor);

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