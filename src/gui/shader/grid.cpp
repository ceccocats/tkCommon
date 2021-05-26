#include "tkCommon/gui/shader/grid.h"

int tk::gui::shader::grid::users = 0;

tk::gui::shader::grid::grid(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/grid.vert";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/grid.frag";        
    shader.init(vertex, fragment);
}

void
tk::gui::shader::grid::close(){
    grid::users--;
    if(grid::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::grid::draw(glm::mat4& modelview, float dim, int n){

    shader.use();

    shader.setMat4("modelview", modelview);
    shader.setInt("n", n);
    shader.setFloat("dim", dim/2);

    shader.setInt("dir", 0);
    glDrawArrays(GL_LINES, 0, n*2 +2);
    
    shader.setInt("dir", 1);
    glDrawArrays(GL_LINES, 0, n*2 +2);

    shader.unuse();

    glCheckError();
}