#pragma once

#include <string>

#include "tkCommon/common.h"
//#include "tkCommon/gui/utils/Shader.h"
#include "tkCommon/gui/utils/Buffer.h"
#include "tkCommon/gui/utils/Texture.h"
#include "tkCommon/gui/utils/Color.h"
//#include <glm/glm.hpp>

void inline glCheckError_(const char *file, int line)
{
    GLenum errorCode;
    while ((errorCode = glGetError()) != GL_NO_ERROR)
    {
        std::string error;
        switch (errorCode)
        {
            case GL_INVALID_ENUM:                  error = "INVALID_ENUM"; break;
            case GL_INVALID_VALUE:                 error = "INVALID_VALUE"; break;
            case GL_INVALID_OPERATION:             error = "INVALID_OPERATION"; break;
            case GL_STACK_OVERFLOW:                error = "STACK_OVERFLOW"; break;
            case GL_STACK_UNDERFLOW:               error = "STACK_UNDERFLOW"; break;
            case GL_OUT_OF_MEMORY:                 error = "OUT_OF_MEMORY"; break;
            case GL_INVALID_FRAMEBUFFER_OPERATION: error = "INVALID_FRAMEBUFFER_OPERATION"; break;
        }
        tkERR(error + " at " + file +":"+std::to_string(line)+":0\n");
    }
}
#define glCheckError() glCheckError_(__FILE__, __LINE__)


namespace tk { namespace gui { namespace common {

struct vertex_t{
    tk::common::Vector3<float>  position;
    tk::common::Vector3<float>  normal;
    tk::common::Vector2<float>  texCoord;
};

struct mesh_t{
    std::vector<unsigned int>   indices;
    std::vector<vertex_t>       vertices;
    Color_t                     color;

    float* vertexBufferPositionNormal(int* n){

        (*n) = vertices.size() * 6;
        float* temp = new float[(*n)];

        for(int i = 0; i < vertices.size(); i++){
            temp[i * 6 + 0] = vertices[i].position.x();
            temp[i * 6 + 1] = vertices[i].position.y();
            temp[i * 6 + 2] = vertices[i].position.z();
            temp[i * 6 + 3] = vertices[i].normal.x();
            temp[i * 6 + 4] = vertices[i].normal.y();
            temp[i * 6 + 5] = vertices[i].normal.z();
        }
        return temp;
    }

    float* vertexBufferPositionTextcoord(int* n){

        (*n) = vertices.size() * 5;
        float* temp = new float[(*n)];

        for(int i = 0; i < vertices.size(); i++){
            temp[i * 5 + 0] = vertices[i].position.x();
            temp[i * 5 + 1] = vertices[i].position.y();
            temp[i * 5 + 2] = vertices[i].position.z();
            temp[i * 5 + 3] = vertices[i].texCoord.x();
            temp[i * 5 + 4] = vertices[i].texCoord.y();
        }
        return temp;
    }
};

struct object3D_t {
    std::vector<mesh_t> meshes;
};


uint8_t* loadImage(std::string filename, int* width, int* height, int* channels);
void     writeImagePng(const std::string&, void* data, const int& width, const int& height, const int& channels);
bool     loadOBJ(std::string filename, object3D_t &obj);

}}}