#include "tkCommon/gui/shader/text.h"

int tk::gui::shader::text::users = 0;

tk::gui::shader::text::text(){

    std::string font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf";

    if(FT_Init_FreeType(&ft)){
        tkERR("Error creating freetype\n");
        return;
    }

    if (FT_New_Face(ft, font.c_str(), 0, &face)){
        tkERR("Failed to load font\n");
        return;
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
    shader.init(vertex, fragment);
}

void
tk::gui::shader::text::close(){
    text::users--;
    if(text::users == 0){
        for (auto const& s : Characters)
            glDeleteTextures(1,&s.second.TextureID);
        shader.close();
    }
}

void 
tk::gui::shader::text::draw(glm::mat4& modelview, std::string text, float targetH, tk::gui::Color_t color){
    float scale = targetH / maxH;
    
    std::memcpy(glm::value_ptr(textColor), color.color, sizeof(textColor));

    shader.use();
    shader.setVec4("textColor", textColor);
    shader.setMat4("projection", modelview);

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
    shader.unuse();
    
    glCheckError();
}