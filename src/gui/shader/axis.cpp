#include "tkCommon/gui/shader/axis.h"

int tk::gui::shader::axis::users = 0;

tk::gui::shader::axis::axis(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.vert";
    std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.geom";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.frag";
    
    shader.init(vertex, fragment, geometry);
}

void
tk::gui::shader::axis::close(){
    axis::users--;
    if(axis::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::axis::draw(glm::mat4& modelview, float width, float height){
            
    shader.use();
    shader.setMat4("modelview", modelview);
    shader.setInt("width",width);
    shader.setInt("height",height);

    glLineWidth(3.0);
    glDrawArrays(GL_POINTS, 0, 1);
    glLineWidth(1.0);
    
    shader.unuse();

    glCheckError();
}