#pragma once
/**
 * @file    pointcloud4f.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that draw pointcloud formed by 4f with shaders
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include "tkCommon/gui/shader/generic.h"
#include <dirent.h>

namespace tk { namespace gui { namespace shader {

/**
 * @brief class that draw a pointcloud organized in 3 points [X Y Z 1 X Y Z 1...]
 * 
 */
class pointcloud4fFeatures  : public tk::gui::shader::generic
{
    private:
        std::map<std::string,tk::gui::Shader*> shaders;

    public:
        std::vector<std::string> colormaps;

        pointcloud4fFeatures(){
            std::string vertex      = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloud4fFeatures.vert";
            std::string geometry    = "";

            std::string searchPath  = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/colormaps/";
            std::string fragPath    = std::string(tkCommon_PATH) + "include/tkCommon/gui/shader/glsl/pointcloudFrag/";
            DIR *dir;
            struct dirent *ent;
            if ((dir = opendir (searchPath.c_str())) != NULL) {
                while ((ent = readdir (dir)) != NULL) {
                    std::string filename    = ent->d_name;
                    if(filename.size() < 4)
                        continue;

                    std::string name        = filename.substr(0,filename.size()-5);
                    std::string fragment    = "pointcloudFrag_"+filename;
                    colormaps.push_back(name);

                    shaders[name] = new tk::gui::Shader();
                    if(shaders[name]->init(vertex,fragPath+fragment,geometry) == false){
                        return;
                    }
                }
                closedir (dir);
            } else {
                clsErr("Directory not exists\n")
                return;
            }           

            vertexPointer.resize(2);
            vertexPointer[0] = {4,4,0};
            vertexPointer[1] = {0,0,0};
        }

        ~pointcloud4fFeatures(){

        }

        void draw(std::string name, tk::gui::Buffer<float>* buffer, int n, float min, float max){

		    glGetFloatv(GL_MODELVIEW_MATRIX, glm::value_ptr(modelview)); 
            vertexPointer[1] = {1,1,n*4};
            buffer->setVertexAttribs(vertexPointer);

            shaders[name]->use();
            shaders[name]->setMat4("modelview",modelview);
            shaders[name]->setFloat("minFeature",min);
            shaders[name]->setFloat("maxFeature",max);

            buffer->use();
            glDrawArrays(GL_POINTS, 0, n);
            buffer->unuse();

            shaders[name]->unuse();

            glCheckError();
        }

        bool close(){
            for (auto const& s : shaders)
                delete s.second;
            return true;
        }
};

}}}