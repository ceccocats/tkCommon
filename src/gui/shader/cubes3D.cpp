#include "tkCommon/gui/shader/cubes3D.h"

int tk::gui::shader::cubes3D::users = 0;

tk::gui::shader::cubes3D::cubes3D(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.vert";
    std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.geom";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/cubes3D.frag";
    
    shader.init(vertex, fragment, geometry);

    //Xcenter. Ycenter, Zcenter
    vertexPointer.push_back({3,13,0});
    //Xsize, Ysize, Zsize
    vertexPointer.push_back({3,13,3});
    //roll, pitch, yaw
    vertexPointer.push_back({3,13,6});
    //red, green, blue, alpha
    vertexPointer.push_back({4,13,9});
}

void
tk::gui::shader::cubes3D::close(){
    cubes3D::users--;
    if(cubes3D::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::cubes3D::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3& lightPos){

    buffer->setVertexAttribs(vertexPointer);

    shader.use();
    shader.setMat4("modelview",modelview);
    shader.setVec3("lightPos", lightPos);

    buffer->use();
    glDrawArrays(GL_POINTS, 0, n);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}