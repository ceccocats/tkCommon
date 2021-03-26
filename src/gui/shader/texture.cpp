#include "tkCommon/gui/shader/texture.h"

int tk::gui::shader::texture::users = 0;

tk::gui::shader::texture::texture(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.frag";
    
    shader.init(vertex, fragment, geometry);

    vertexPointer.push_back({3,5,0});
    vertexPointer.push_back({2,5,3});
}

void
tk::gui::shader::texture::close(){
    texture::users--;
    if(texture::users == 0){
        shader.close();
    }
}