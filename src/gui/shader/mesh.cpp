#include "tkCommon/gui/shader/mesh.h"

int tk::gui::shader::mesh::users = 0;

tk::gui::shader::mesh::mesh(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.vert";
    std::string geometry    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.geom";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/mesh.frag";
    
    bool status = shader.init(vertex, fragment, geometry);
    if(status == false) return;

    vertexPointer.push_back({3,18,0});
    vertexPointer.push_back({3,18,3});
    vertexPointer.push_back({3,18,6});
    vertexPointer.push_back({3,18,9});
    vertexPointer.push_back({3,18,12});
    vertexPointer.push_back({3,18,15});
    return;
}

void
tk::gui::shader::mesh::close(){
    mesh::users--;
    if(mesh::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::mesh::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3 lightPos, tk::gui::Color_t color, float ambientStrength, bool useLight){

    buffer->setVertexAttribs(vertexPointer);

    std::memcpy(glm::value_ptr(meshColor), color.color, sizeof(meshColor));

    shader.use();
    shader.setMat4("modelview",modelview);
    shader.setVec4("color", meshColor);
    shader.setVec3("lightPos",lightPos);
    shader.setFloat("ambientStrength",ambientStrength);
    shader.setBool("useLight",useLight);

    buffer->use();
    glDrawArrays(GL_POINTS, 0, n);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}