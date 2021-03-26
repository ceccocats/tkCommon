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
        static int users;
        
        mesh();
    public:
        static mesh* getInstance(){
            static mesh instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3 lightPos, 
            tk::gui::Color_t color = tk::gui::color::WHITE, float ambientStrength = 0.6, bool useLight = true);
        void close();
};

}}}