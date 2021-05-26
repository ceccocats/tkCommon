#pragma once
/**
 * @file    axisPlot.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw axis
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z ROLL PITCH YAW ...]
 * 
 */
class axisPlot : public tk::gui::shader::generic
{
    private:
        axisPlot();
        static int users;

    public:
        static axisPlot* getInstance(){
            static axisPlot instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview,tk::gui::Buffer<float>* buffer, int n, float size = 1.0f);
        void close();
};

}}}