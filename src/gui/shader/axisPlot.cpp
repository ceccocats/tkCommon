#include "tkCommon/gui/shader/axisPlot.h"

int tk::gui::shader::axisPlot::users = 0;

tk::gui::shader::axisPlot::axisPlot(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axisPlot.vert";
    std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axisPlot.geom";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/axis.frag";
    
    shader.init(vertex, fragment, geometry);
    vertexPointer.resize(2);
}

void
tk::gui::shader::axisPlot::close(){
    axisPlot::users--;
    if(axisPlot::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::axisPlot::draw(glm::mat4& modelview,tk::gui::Buffer<float>* buffer, int n, float size){

    vertexPointer[0] = {3,3,0};
    vertexPointer[1] = {3,3,n*3};
    buffer->setVertexAttribs(vertexPointer);

    shader.use();
    shader.setMat4("modelview",modelview);

    buffer->use();
    glLineWidth(size);
    glDrawArrays(GL_POINTS, 0, n);
    glLineWidth(1.0);
    buffer->unuse();
    
    shader.unuse();

    glCheckError();
}