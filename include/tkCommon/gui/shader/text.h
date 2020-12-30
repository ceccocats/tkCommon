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

    public:
        bool init(std::string font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"){

            if(FT_Init_FreeType(&ft)){
                tkERR("Error creating freetype\n");
                return false;
            }

            if (FT_New_Face(ft, font.c_str(), 0, &face)){
                tkERR("Failed to load font\n");
                return false;
            }

            FT_Set_Pixel_Sizes(face, 0, 48);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

            for (unsigned char c = 0; c < 128; c++){

                // Load character glyph 
                if (FT_Load_Char(face, c, FT_LOAD_RENDER)){
                    tkERR("Failed to load character id"+std::to_string(c)+"\n");
                    continue;
                }
                // generate texture
                unsigned int texture;
                glGenTextures(1, &texture);
                glBindTexture(GL_TEXTURE_2D, texture);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RED, face->glyph->bitmap.width, face->glyph->bitmap.rows,
                                0, GL_RED, GL_UNSIGNED_BYTE, face->glyph->bitmap.buffer );
                // set texture options
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                // now store character for later use
                Character character = {
                    texture,
                    glm::ivec2(face->glyph->bitmap.width, face->glyph->bitmap.rows),
                    glm::ivec2(face->glyph->bitmap_left, face->glyph->bitmap_top),
                    static_cast<unsigned int>(face->glyph->advance.x)
                };
                if(character.Size.y > maxH) {
                    maxH = character.Size.y;
                }
                Characters.insert(std::pair<char, Character>(c, character));
            }
            glBindTexture(GL_TEXTURE_2D, 0);

            // destroy FreeType once we're finished
            FT_Done_Face(face);
            FT_Done_FreeType(ft);        

            // configure VAO/VBO for texture quads
            glGenVertexArrays(1, &VAO);
            glGenBuffers(1, &VBO);
            glBindVertexArray(VAO);
            glBindBuffer(GL_ARRAY_BUFFER, VBO);
            glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 6 * 4, NULL, GL_DYNAMIC_DRAW);
            glEnableVertexAttribArray(0);
            glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), 0);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glBindVertexArray(0);

            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/text.vert";
            std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/text.frag";        
            return shader.init(vertex, fragment);
        }

        void draw(std::string text, float targetH = 0.5, tk::gui::Color_t color = tk::gui::color::WHITE){
            float scale = targetH / maxH;
            
            std::memcpy(glm::value_ptr(textColor), color.color, sizeof(textColor));
            glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(projection)); 

            shader.use();
            shader.setVec4("textColor", textColor);
            shader.setMat4("projection", projection);

            glActiveTexture(GL_TEXTURE0);
            glBindVertexArray(VAO);

            float x=0, y=0;
            // iterate through all characters
            std::string::const_iterator c;
            for (c = text.begin(); c != text.end(); c++){

                Character ch = Characters[*c];

                float xpos = x + ch.Bearing.x * scale;
                float ypos = y - (ch.Size.y - ch.Bearing.y) * scale;

                float w = ch.Size.x * scale;
                float h = ch.Size.y * scale;

                // update VBO for each character
                float vertices[6][4] = {
                    { xpos,     ypos + h,   0.0f, 0.0f },            
                    { xpos,     ypos,       0.0f, 1.0f },
                    { xpos + w, ypos,       1.0f, 1.0f },

                    { xpos,     ypos + h,   0.0f, 0.0f },
                    { xpos + w, ypos,       1.0f, 1.0f },
                    { xpos + w, ypos + h,   1.0f, 0.0f }           
                };

                // render glyph texture over quad
                glBindTexture(GL_TEXTURE_2D, ch.TextureID);
                glBindBuffer(GL_ARRAY_BUFFER, VBO);
                glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vertices), vertices);
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                x += (ch.Advance >> 6) * scale;
            }
            glBindVertexArray(0);
            glBindTexture(GL_TEXTURE_2D, 0);
            
            glCheckError();
        }

        bool close(){
            return shader.close();
        }
};

}}}