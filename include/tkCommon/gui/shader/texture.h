#pragma once
/**
 * @file    texture.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw a texture with shader
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a texture, data organized in [X Y Z textX textY ...]
 * 
 */
class texture : public tk::gui::shader::generic
{
    private:
        static int users;
        texture();

    public:
        static texture* getInstance(){
            static texture instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, tk::gui::Texture<uint8_t>* text, tk::gui::Buffer<float>* buffer, int triangles);
        void close();
};

}}}