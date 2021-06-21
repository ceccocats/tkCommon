#include "tkCommon/gui/shader/simpleMesh.h"

tk::gui::shader::simpleMesh::simpleMesh() 
{
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/simpleMesh.vert";
    std::string geometry    = ""; //std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/simpleMesh.geom";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/simpleMesh.frag";

    bool status = shader.init(vertex, fragment, geometry);
    if(status == false) return;

    vertexPointer.push_back({3,6,0}); // vec3 vPos
    vertexPointer.push_back({3,6,3}); // vec3 normal

    users = 0;
}


void 
tk::gui::shader::simpleMesh::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, glm::vec3 lightPos, bool useLight, tk::gui::Color_t color) 
{
    buffer->setVertexAttribs(vertexPointer);

    std::memcpy(glm::value_ptr(pointColor), color.color, sizeof(pointColor));

    shader.use();
    shader.setMat4("modelview", modelview);
    shader.setVec4("color", pointColor);
    shader.setBool("useLight", useLight);
    shader.setVec3("lightPos", lightPos);
    shader.setFloat("ambientStrength", 0.8f);

    buffer->use();
    glDrawArrays(GL_TRIANGLES, 0, n);
    buffer->unuse();

    shader.unuse();
    glCheckError();
}

void 
tk::gui::shader::simpleMesh::draw(glm::mat4& modelview, const std::vector<tk::gui::Buffer<float>> &vectorBuffer, const std::vector<glm::mat4> &poses, glm::vec3 lightPos, bool useLight, tk::gui::Color_t color)
{
    std::memcpy(glm::value_ptr(pointColor), color.color, sizeof(pointColor));

    shader.use();
    shader.setVec4("color", pointColor);
    shader.setBool("useLight", useLight);
    shader.setVec3("lightPos", lightPos);
    shader.setFloat("ambientStrength", 0.8f);

    for (int i = 0; i < vectorBuffer.size(); ++i) {
        shader.setMat4("modelview", modelview * poses[i]);

        vectorBuffer[i].setVertexAttribs(vertexPointer);
        vectorBuffer[i].use();
        glDrawArrays(GL_TRIANGLES, 0, vectorBuffer[i].size()/6);
        vectorBuffer[i].unuse();
    }
    
    shader.unuse();
    glCheckError();
}

void 
tk::gui::shader::simpleMesh::close()
{
    users--;
    if(users == 0){
        shader.close();
    }
}