#pragma once
/**
 * @file    grid.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw text
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"

#include <freetype2/ft2build.h>
#include FT_FREETYPE_H

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw grid
 * 
 */
class text : public tk::gui::shader::generic
{
    private:

        FT_Library  ft;
        FT_Face     face;

        struct Character {
            unsigned int TextureID; // ID handle of the glyph texture
            glm::ivec2   Size;      // Size of glyph
            glm::ivec2   Bearing;   // Offset from baseline to left/top of glyph
            unsigned int Advance;   // Horizontal offset to advance to next glyph
        };
        std::map<GLchar, Character> Characters;

        unsigned int VAO, VBO;
        glm::mat4 projection;

        glm::vec4 textColor;
        float maxH = 0;

        static int users;
        text();

    public:
        static text* getInstance(){
            static text instance;
            users++;
            return &instance;
        }
        void draw(glm::mat4& modelview, std::string text, float targetH = 0.5, tk::gui::Color_t color = tk::gui::color::WHITE);
        void close();
};

}}}