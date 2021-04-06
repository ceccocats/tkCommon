#include "tkCommon/gui/utils/CommonViewer.h"
#include "tkCommon/gui/utils/stb_image.h"
#include "tkCommon/gui/utils/OBJ_Loader.h"

namespace tk { namespace gui { namespace common {

uint8_t* loadImage(std::string filename, int* width, int* height, int* channels){

    uint8_t* data = stbi_load(filename.c_str(),width,height,channels,0);
    if(data == NULL){
        tkERR(std::string{"Error opening: "+filename+"\n"});
    }
    return data;
}

void copyVertexFromOBJL(vertex_t *tkv, objl::Vertex* v){
    tkv->position.x() = v->Position.X;
    tkv->position.y() = v->Position.Y;
    tkv->position.z() = v->Position.Z;

    tkv->normal.x() = v->Normal.X;
    tkv->normal.y() = v->Normal.Y;
    tkv->normal.z() = v->Normal.Z;

    tkv->texCoord.x() = v->TextureCoordinate.X;
    tkv->texCoord.y() = 1 - v->TextureCoordinate.Y;
}

bool loadOBJ(std::string filename, object3D_t &obj) {
    
    tkMSG(std::string{"loading: "+filename+"\n"});

    //Load and check
    objl::Loader loader;
    bool status = loader.LoadFile(filename.c_str());
    if(status == false){
         tkERR(std::string{"Error opening: "+filename+"\n"});
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
            copyVertexFromOBJL(&vertex, &loader.LoadedMeshes[i].Vertices[j]);
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