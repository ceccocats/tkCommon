#pragma once

#include <string>
#include "tkCommon/gui/Color.h"
#include "tkCommon/common.h"

namespace tk { namespace gui { namespace common {

struct vertex_t{
    tk::common::Vector3<float>  position;
    tk::common::Vector3<float>  normal;
    tk::common::Vector2<float>  texCoord;
};

struct mesh_t{
    std::vector<unsigned int>   indices;
    std::vector<vertex_t>       vertices;
    tk::gui::Color_t            color;

    float* vertexBufferPositionNormal(int* n){

        (*n) = vertices.size() * 6;
        float* temp = new float[(*n)];

        for(int i = 0; i < vertices.size(); i++){
            temp[i * 6 + 0] = vertices[i].position.x;
            temp[i * 6 + 1] = vertices[i].position.y;
            temp[i * 6 + 2] = vertices[i].position.z;
            temp[i * 6 + 3] = vertices[i].normal.x;
            temp[i * 6 + 4] = vertices[i].normal.y;
            temp[i * 6 + 5] = vertices[i].normal.z;
        }
        return temp;
    }

    float* vertexBufferPositionTextcoord(int* n){

        (*n) = vertices.size() * 5;
        float* temp = new float[(*n)];

        for(int i = 0; i < vertices.size(); i++){
            temp[i * 5 + 0] = vertices[i].position.x;
            temp[i * 5 + 1] = vertices[i].position.y;
            temp[i * 5 + 2] = vertices[i].position.z;
            temp[i * 5 + 3] = vertices[i].texCoord.x;
            temp[i * 5 + 4] = vertices[i].texCoord.y;
        }
        return temp;
    }
};

struct object3D_t {
    std::vector<mesh_t> meshes;
};


uint8_t* loadImage(std::string filename, int* width, int* height, int* channels);
bool     loadOBJ(std::string filename, object3D_t &obj);

}}}