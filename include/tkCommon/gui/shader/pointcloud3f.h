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
        static int users;

        pointcloud3f();

    public:
        static pointcloud3f* getInstance(){
            static pointcloud3f instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, tk::gui::Color_t color = tk::gui::color::WHITE);
        void close();
};

}}}