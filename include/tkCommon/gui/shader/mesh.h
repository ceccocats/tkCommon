#pragma once
/**
 * @file    mash.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw mesh formed by couples of triangles and normals
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a mesh organized in 18 points an normals (or you can use index vector)
 *          [x0 y0 z0 nx0 ny0 nz0 x1 y1 z1 nx1 ny1 nz1 x2 y2 z2 nx2 ny2 nz2]
 * 
 *                  (x1,y1,z1)
 *                (nx1, ny1, nz1)
 *                      *
 *                     / \
 *                    /   \
 *                   /     \
 *       (x0,y0,z0) *-------* (x2,y2,z2)
 *    (nx0, ny0, nz0)       (nx2, ny2, nz2)
 * 
 */
class mesh : public tk::gui::shader::generic
{
    private:
        glm::vec4 meshColor;

    public:
        
        mesh(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.vert";
            std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.geom";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.frag";
            
            bool status = shader.init(vertex, fragment, geometry);
            if(status == false) return;

            vertexPointer.push_back({3,18,0});
		    vertexPointer.push_back({3,18,3});
		    vertexPointer.push_back({3,18,6});
		    vertexPointer.push_back({3,18,9});
		    vertexPointer.push_back({3,18,12});
		    vertexPointer.push_back({3,18,15});
            return;
        }

        ~mesh(){

        }

        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3 lightPos, tk::gui::Color_t color = tk::gui::color::WHITE, float ambientStrength = 0.6, bool useLight = true){

            buffer->setVertexAttribs(vertexPointer);

            std::memcpy(glm::value_ptr(meshColor), color.color, sizeof(meshColor));

            shader.use();
            shader.setMat4("modelview",modelview);
            shader.setVec4("color", meshColor);
            shader.setVec3("lightPos",lightPos);
            shader.setFloat("ambientStrength",ambientStrength);
            shader.setBool("useLight",useLight);

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