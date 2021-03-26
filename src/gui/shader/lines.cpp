#include "tkCommon/gui/shader/lines.h"

int tk::gui::shader::lines::users = 0;

tk::gui::shader::lines::lines(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/lines.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/lines.frag";
    
    shader.init(vertex, fragment, geometry);
    
    vertexPointer.resize(2);
    vertexPointer[0] = {3,7,0};
    vertexPointer[1] = {4,7,3};
}

void
tk::gui::shader::lines::close(){
    lines::users--;
    if(lines::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::lines::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, float size, GLenum linemode){

    buffer->setVertexAttribs(vertexPointer);

    shader.use();
    shader.setMat4("modelview",modelview);

    buffer->use();
    glLineWidth(size);
    glDrawArrays(linemode, 0, n);
    glLineWidth(1.0f);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}