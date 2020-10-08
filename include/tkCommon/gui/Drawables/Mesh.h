#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/mesh.h"

namespace tk{ namespace gui{

	class Mesh : public Drawable {

        private:
            std::vector<tk::gui::Buffer<float>> obj;
            std::vector<tk::gui::Color_t> objColors;
            std::string filename;

        public:

            Mesh(std::string filename){
                this->filename = filename;
            }

            ~Mesh(){

            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::mesh();

                tk::gui::common::object3D_t objMesh;
                tk::gui::common::loadOBJ(filename, objMesh);
                obj.resize(objMesh.meshes.size());
                objColors.resize(objMesh.meshes.size());
                for(int i = 0; i < objMesh.meshes.size(); i++){

                    obj[i].init();

                    float* data; int n;
                    data = objMesh.meshes[i].vertexBufferPositionNormal(&n);
                    obj[i].setData(data,n);
                    delete[] data;

                    //fill indices
                    obj[i].setIndexVector(objMesh.meshes[i].indices.data(),objMesh.meshes[i].indices.size());
                    objColors[i] = objMesh.meshes[i].color;
                }
                objMesh.meshes.clear();
            }

            void draw(tk::gui::Viewer *viewer){
                tk::gui::shader::mesh* shaderMesh= (tk::gui::shader::mesh*) shader;

                for(int i = 0; i < obj.size(); i++){	
                    shaderMesh->draw(&obj[i], obj[i].size(), viewer->lightPos, objColors[i]);
                }		
            }

            void imGuiSettings(){
                
            }

            void imGuiInfos(){
                
            }

            void beforeDraw(){
                
            }

            void onClose(){
                tk::gui::shader::mesh* shaderMesh= (tk::gui::shader::mesh*) shader;
                shaderMesh->close();
                delete shaderMesh;
            }

            std::string toString(){
                return filename.substr(filename.find_last_of("/\\")+1);
            }
	};
}}