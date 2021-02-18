#pragma once
/**
 * @file    pointcloudRGBA.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw pointcloud formed by 4f with shaders
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z 1 X Y Z 1...]
 * 
 */
class pointcloudRGBA  : public tk::gui::shader::generic
{
    public:
        pointcloudRGBA(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudRGBA.vert";
            std::string geometry    = "";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudRGBA.frag";
            
            shader.init(vertex, fragment, geometry);

            vertexPointer.resize(4);
            vertexPointer[0] = {4,4,0};
        }

        ~pointcloudRGBA(){

        }

        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, bool red, float r_min, float r_max, bool green, float g_min, float g_max, bool blue, float b_min, float b_max, float alpha = 1.0f){

            if(red == true){
                vertexPointer[1] = {1,1,n*4};
                if(green == true){
                    vertexPointer[2] = {1,1,n*5};
                    if(blue == true){
                        vertexPointer[3] = {1,1,n*6};
                    }else{
                        vertexPointer[3] = {1,1,0};
                    }
                }else{
                     vertexPointer[2] = {1,1,0};
                     vertexPointer[3] = {1,1,0};
                }
            }else{
                return;
            }
            buffer->setVertexAttribs(vertexPointer);

            shader.use();
            shader.setFloat("alpha",alpha);
            shader.setMat4("modelview",modelview);
            shader.setBool("red",red);
            if(red == true){
                shader.setFloat("r_min",r_min);
                shader.setFloat("r_max",r_max);
            }
            shader.setBool("green",green);
            if(green == true){
                shader.setFloat("g_min",g_min);
                shader.setFloat("g_max",g_max);
            }
            shader.setBool("blue",blue);
            if(blue == true){
                shader.setFloat("b_min",b_min);
                shader.setFloat("b_max",b_max);
            }

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