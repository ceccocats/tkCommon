#pragma once
/**
 * @file    pointcloud4f.h
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
class pointcloud4f  : public tk::gui::shader::generic
{
    private:
        glm::vec4 pointColor;
        static int users;

        pointcloud4f();

    public:
       static pointcloud4f* getInstance(){
            static pointcloud4f instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, tk::gui::Color_t color = tk::gui::color::WHITE);
        void close();
};

}}}