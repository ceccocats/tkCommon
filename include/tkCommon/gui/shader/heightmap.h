#pragma once
/**
 * @file    stripline.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw a heightmap
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a heightmap
 * 
 */
class heightmap : public tk::gui::shader::generic
{
    private:
        tk::gui::Buffer<float> map;
        tk::math::Mat<unsigned int> index;
        int prec = 0;
        
        void calculateIndex(int rows, int cols);

        heightmap();
        static int users;

    public:
        static heightmap* getInstance(){
            static heightmap instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::math::Mat<float>& cords, tk::math::Mat<float>& colors, int rows, int cols);
        void close();
};

}}}