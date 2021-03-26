#pragma once
/**
 * @file    grid.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw grid
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw grid
 * 
 */
class grid : public tk::gui::shader::generic
{
    private:
        grid();
        static int users;
    public:  
        static grid* getInstance(){
            static grid instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, float dim = 1.0f, int n = 50);
        void close();
};

}}}