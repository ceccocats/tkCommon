#include "tkCommon/gui/shader/pointcloud3f.h"

int tk::gui::shader::pointcloud3f::users = 0;


tk::gui::shader::pointcloud3f::pointcloud3f(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloud3f.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudFrag/pointcloud_uniformColor.frag";
    
    shader.init(vertex, fragment, geometry);

    vertexPointer.push_back({3,3,0});
}

void
tk::gui::shader::pointcloud3f::close(){
    pointcloud3f::users--;
    if(pointcloud3f::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::pointcloud3f::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, tk::gui::Color_t color){

    buffer->setVertexAttribs(vertexPointer);

    std::memcpy(glm::value_ptr(pointColor), color.color, sizeof(pointColor));

    shader.use();
    shader.setMat4("modelview",modelview);
    shader.setVec4("color",pointColor);

    buffer->use();
    glDrawArrays(GL_POINTS, 0, n);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}