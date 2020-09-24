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

#include "tkCommon/gui/tkShader.h"
#include "tkCommon/gui/tkBufferGl.h"
#include "tkCommon/gui/Color.h"

namespace tk { namespace gui { namespace components {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z X Y Z...]
 * 
 */
class pointcloud3f
{
    private:
        std::vector<tk::gui::vertexAttribs_t>   vertexPointer;
        tk::gui::tkShader                       shader;

        glm::mat4                               modelview;

    public:
        bool init(){
            std::string vertex      = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/pointcloud3f/pointcloud3f.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(TKPROJ_PATH) + "include/tkCommon/gui/components/pointcloud3f/pointcloud3f.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false)
                return false;

            vertexPointer.push_back({3,3,0});
            return true;
        }

        void draw(tk::gui::tkBufferGl<float>* buffer, int n, tk::gui::Color_t color = tk::gui::color::WHITE){

		    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 
            buffer->setVertexAttribs(vertexPointer);

            glm::vec4 pointColor(   float(color.r)/255.0f,
                                    float(color.g)/255.0f,
                                    float(color.b)/255.0f,
                                    float(color.a)/255.0f    
                                );
            shader.use();
            shader.setMat4("modelview",modelview);
            shader.setVec4("color",pointColor);

            buffer->use();
            glDrawArrays(GL_POINTS, 0, n);
            buffer->unuse();

            shader.unuse();
        }

        bool close(){
            return shader.close();
        }


};

}}}