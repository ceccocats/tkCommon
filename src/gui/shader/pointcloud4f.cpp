#include "tkCommon/gui/shader/pointcloud4f.h"

int tk::gui::shader::pointcloud4f::users = 0;

tk::gui::shader::pointcloud4f::pointcloud4f(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloud.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudFrag/pointcloud_uniformColor.frag";
    
    shader.init(vertex, fragment, geometry);
    vertexPointer.push_back({4,4,0});
}

void
tk::gui::shader::pointcloud4f::close(){
    pointcloud4f::users--;
    if(pointcloud4f::users == 0){
        shader.close();
    }
}

void 
tk::gui::shader::pointcloud4f::draw(glm::mat4& modelview, tk::gui::Buffer<float>* buffer, int n, tk::gui::Color_t color){

    if(n == 0){
        //tkWRN("Pontcloud has 0 points\n")
        return;
    }

    buffer->setVertexAttribs(vertexPointer);

    std::memcpy(glm::value_ptr(pointColor), color.color, sizeof(pointColor));

    shader.use();;
    shader.setMat4("modelview",modelview);
    shader.setVec4("color",pointColor);

    buffer->use();
    glDrawArrays(GL_POINTS, 0, n);
    buffer->unuse();

    shader.unuse();

    glCheckError();
}