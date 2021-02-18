#pragma once
/**
 * @file    tkShader.h
 * @author  Luca Bartoli, Fabio bagni, Gatti Francesco, Massimiliano bosi (you@domain.com)
 * @brief   class that handle compiling, linking and running shaders
 * @version 1.0
 * @date    2020-07-10
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#include <GL/glew.h>
#include <GLFW/glfw3.h> 

#include <GL/glew.h> 
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <map>

#include "tkCommon/exceptions.h"

namespace tk { namespace gui {

class Shader
{
    public:
        /**
         *  init method 
         * 
         * @param vertexPath    path to vertex shader
         * @param fragmentPath  path to fragment shader
         * @param geometryPath  path to geometry shader (not necessary)
         * 
         */
        bool init(std::string vertexPath, std::string fragmentPath, std::string geometryPath = "");

        /**
         *  use method
         */
        void use();

        /**
         *  unuse method
         */
        void unuse();

        /**
         *  close method
         */
        bool close();

        /**
         *  utility uniform functions
         */
        // ------------------------------------------------------------------------
        void setBool(const std::string &name, bool value);
        void setInt(const std::string &name, int value);
        void setFloat(const std::string &name, float value);
        void setVec2(const std::string &name, const glm::vec2 &value);
        void setVec2(const std::string &name, float x, float y);
        void setVec3(const std::string &name, const glm::vec3 &value);
        void setVec3(const std::string &name, float x, float y, float z);
        void setVec4(const std::string &name, const glm::vec4 &value);
        void setVec4(const std::string &name, float x, float y, float z, float w);
        void setMat2(const std::string &name, const glm::mat2 &mat);
        void setMat3(const std::string &name, const glm::mat3 &mat);
        void setMat4(const std::string &name, const glm::mat4 &mat);
        // ------------------------------------------------------------------------

    private:

        unsigned int ID;
        std::map<std::string,int> map; 

        /**
         *  private method for check compile error
         */
        void checkCompileErrors(GLuint shader, std::string type, std::string filename);

        /**
         *  private method for load source shader code and processing include directives
         */
        std::string load(std::string path);
};

}}