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
    private:
        static int users;
        pointcloudRGBA();

    public:
        static pointcloudRGBA* getInstance(){
            static pointcloudRGBA instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, bool red, float r_min, float r_max, bool green, 
            float g_min, float g_max, bool blue, float b_min, float b_max, float alpha = 1.0f);
        void close();
};

}}}