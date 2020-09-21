#pragma once

#include <string>
#include "tkCommon/gui/stb_image.h"
#include "tkCommon/gui/OBJ_Loader.h"

namespace tk { namespace gui { namespace common {

struct vertex_t{
    tk::common::Vector3<float>  position;
    tk::common::Vector3<float>  normal;
    tk::common::Vector2<float>  texCoord;

    void copyFrom(objl::Vertex* v){
        this->position.x = v->Position.X;
        this->position.y = v->Position.Y;
        this->position.z = v->Position.Z;

        this->normal.x = v->Normal.X;
        this->normal.y = v->Normal.Y;
        this->normal.z = v->Normal.Z;

        this->texCoord.x = v->TextureCoordinate.X;
        this->texCoord.y = 1 - v->TextureCoordinate.Y;
    }
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


uint8_t* loadImage(std::string filename, int* width, int* height, int* channels){

    tk::tformat::printMsg("Viewer",std::string{"loading: "+filename+"\n"});
    uint8_t* data = stbi_load(filename.c_str(),width,height,channels,0);
    if(data == NULL){
        tk::tformat::printErr("Viewer",std::string{"Error opening: "+filename+"\n"});
    }
    return data;
}

bool loadOBJ(std::string filename, object3D_t &obj) {
    
    tk::tformat::printMsg("Viewer",std::string{"loading: "+filename+"\n"});

    //Load and check
    objl::Loader loader;
    bool status = loader.LoadFile(filename.c_str());
    if(status == false){
         tk::tformat::printErr("Viewer",std::string{"Error opening: "+filename+"\n"});
         return false;
    }

    std::setlocale(LC_ALL, "C");

    //Clear obj
    obj.meshes.clear();

    for(int i = 0; i < loader.LoadedMeshes.size(); i++){

        mesh_t mesh;
        for(int j = 0; j < loader.LoadedMeshes[i].Indices.size(); j++){

            int indice  = loader.LoadedMeshes[i].Indices[j];
            mesh.indices.push_back(indice);

            vertex_t vertex;
            vertex.copyFrom(&loader.LoadedMeshes[i].Vertices[j]);
            mesh.vertices.push_back(vertex);
        }

        float r = loader.LoadedMeshes[i].MeshMaterial.Kd.X;
        float g = loader.LoadedMeshes[i].MeshMaterial.Kd.Y;
        float b = loader.LoadedMeshes[i].MeshMaterial.Kd.Z;
        mesh.color = tk::gui::color4f(r,g,b,1);
        obj.meshes.push_back(mesh);
    }

    return true;
}



}}}