#pragma once
/**
 * @file    lines.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw lines from points
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a lines formed by [X Y Z R G B A ...]
 * 
 */
class lines : public tk::gui::shader::generic
{
    private:
        lines();
        static int users;

    public:

        struct line_t{
            float x;
            float y;
            float z;
            float r;
            float g;
            float b;
            float a;
        };

    public:
        static lines* getInstance(){
            static lines instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, float size = 1.0f, GLenum linemode = GL_LINE_STRIP);
        void close();
};

}}}