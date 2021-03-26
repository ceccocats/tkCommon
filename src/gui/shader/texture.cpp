#include "tkCommon/gui/shader/texture.h"

int tk::gui::shader::texture::users = 0;

tk::gui::shader::texture::texture(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/texture.frag";
    
    bool status = shader.init(vertex, fragment, geometry);

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

void 
tk::gui::shader::texture::draw(glm::mat4& modelview, tk::gui::Texture<uint8_t>* text, tk::gui::Buffer<float>* buffer, int triangles){

    buffer->setVertexAttribs(vertexPointer);

    shader.use();
    shader.setMat4("modelview",modelview);
    text->use();
    buffer->use();

    if(buffer->hasEBO()){
        glDrawElements(GL_TRIANGLES, triangles, GL_UNSIGNED_INT, 0);
    }else{
        glDrawArrays(GL_TRIANGLES, 0, triangles);
    }

    buffer->unuse();
    text->unuse();
    shader.unuse();

    glCheckError();
}