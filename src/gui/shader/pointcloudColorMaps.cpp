#include "tkCommon/gui/shader/pointcloudColorMaps.h"

int tk::gui::shader::pointcloudColorMaps::users = 0;


tk::gui::shader::pointcloudColorMaps::pointcloudColorMaps(){
    std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudColorMaps.vert";
    std::string geometry    = "";
    std::string fragment    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudFrag/";

    std::string searchPath  = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/colormaps/";
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir (searchPath.c_str())) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string filename    = ent->d_name;
            if(filename.size() < 11)
                continue;
            
            std::string name    = filename.substr(0,filename.size()-5);
            shaders[name]       = new tk::gui::Shader();


            std::string path    = fragment+"pointcloud_"+filename;                    
            if(shaders[name]->init(vertex,path,geometry) == false){
                return;
            }
        }
        closedir (dir);
    } else {
        tkERR("Directory not exists\n")
        return;
    }     

    //Fill name ordered
    for (auto const& x : shaders){
        colormaps.push_back(x.first);
    }    
}

void
tk::gui::shader::pointcloudColorMaps::close(){
    pointcloudColorMaps::users--;
    if(pointcloudColorMaps::users == 0){
        for (auto const& s : shaders)
            delete s.second;
        shader.close();
    }
}

void 
tk::gui::shader::pointcloudColorMaps::draw(glm::mat4& modelview, std::string name, tk::gui::Buffer<float>* buffer, 
    int nPoints, float minValue, float maxValue, int axis, float alpha){

    tkASSERT(axis == -1 || axis == 0 || axis == 1 || axis == 2);

    if(axis == -1){
        vertexPointer.resize(2);
        vertexPointer[0] = {4,4,0};
        vertexPointer[1] = {1,1,nPoints*4};
    }else{
        vertexPointer.resize(1);
        vertexPointer[0] = {4,4,0};
    }
    buffer->setVertexAttribs(vertexPointer);

    shaders[name]->use();
    shaders[name]->setMat4("modelview",modelview);
    shaders[name]->setFloat("minFeature",minValue);
    shaders[name]->setFloat("maxFeature",maxValue);
    shaders[name]->setFloat("alpha",alpha);
    shaders[name]->setInt("axis",axis);            

    buffer->use();
    glDrawArrays(GL_POINTS, 0, nPoints);
    buffer->unuse();

    shaders[name]->unuse();

    glCheckError();
}