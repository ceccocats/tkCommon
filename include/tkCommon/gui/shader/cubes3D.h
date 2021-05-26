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
    private:
        cubes3D();
        static int users;
    public:
        ~cubes3D();
        static cubes3D* getInstance(){
            static cubes3D instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3& lightPos);
        void close();
};

}}}