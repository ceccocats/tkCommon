#pragma once
/**
 * @file    stripline.h
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
 * @brief class that draw a lines formed by [X Y Z ...]
 * 
 */
class linesMonocolor : public tk::gui::shader::generic
{
    private:
        glm::vec4 linesColor;
        static int users;

        linesMonocolor();

    public:

        struct line_color_t{
            float x;
            float y;
            float z;

            line_color_t& operator =(const line_color_t& a){
                x = a.x;
                y = a.y;
                z = a.z;
                return *this;
            }
        };

    public:
       static linesMonocolor* getInstance(){
            static linesMonocolor instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, int size = 1.0f, 
                tk::gui::Color_t color = tk::gui::color::WHITE, GLenum linemode = GL_LINE_STRIP, int offset = 0);
        void close();
};

}}}