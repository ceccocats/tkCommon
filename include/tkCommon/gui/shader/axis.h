#pragma once
/**
 * @file    axis.h
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
 * @brief class that draw a pointcloud organized in 3 points [X Y Z X Y Z...]
 * 
 */
class axis : public tk::gui::shader::generic
{
    private:
        axis();    
        static int users;
    public:
        static axis* getInstance(){
            static axis instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, float width, float height);
        void close();
};

}}}