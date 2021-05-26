#include "tkCommon/gui/shader/linesMonocolor.h"

int tk::gui::shader::linesMonocolor::users = 0;

tk::gui::shader::linesMonocolor::linesMonocolor(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/linesMonocolor.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/linesMonocolor.frag";
    
    shader.init(vertex, fragment, geometry);

    vertexPointer.resize(1);
}

void
tk::gui::shader::linesMonocolor::close(){
    linesMonocolor::users--;
    if(linesMonocolor::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::linesMonocolor::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, int size, 
        tk::gui::Color_t color, GLenum linemode, int offset){

    vertexPointer[0] = {3,3,offset};
    buffer->setVertexAttribs(vertexPointer);

    std::memcpy(glm::value_ptr(linesColor), color.color, sizeof(linesColor));

    shader.use();
    shader.setMat4("modelview",modelview);
    shader.setVec4("color",linesColor);

    buffer->use();
    glLineWidth(size);
    glDrawArrays(linemode, 0, n);
    glLineWidth(1.0f);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}