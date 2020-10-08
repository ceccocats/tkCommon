#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/mesh.h"

namespace tk{ namespace gui{

	class Mesh : public Drawable {

        private:
            std::vector<tk::gui::Buffer<float>> obj;
            std::vector<tk::gui::Color_t> objColors;
            std::string filename;
            float imgui_color[4];
            tk::gui::Color_t color;

        public:

            Mesh(std::string filename){
                this->filename = filename;

                imgui_color[0]=0;
                imgui_color[1]=0;
                imgui_color[2]=0;
                imgui_color[3]=0;
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

                tk::gui::Color_t meshColor;
                for(int i = 0; i < obj.size(); i++){
                    meshColor.r = objColors[i].r + color.r;
                    meshColor.g = objColors[i].g + color.g;
                    meshColor.b = objColors[i].b + color.b;	
                    meshColor.a = objColors[i].a + color.a;
                    shaderMesh->draw(&obj[i], obj[i].size(), viewer->lightPos, meshColor);
                }		
            }

            void imGuiSettings(){
                ImGui::Text("Setting value to sum in a color mesh");
                ImGui::SliderFloat("R",&imgui_color[0],-1.0f,1.0f,"%.1f");
                ImGui::SameLine();
                if (ImGui::Button("reset R")){
                    imgui_color[0] = 0;
                }
                ImGui::SliderFloat("G",&imgui_color[1],-1.0f,1.0f,"%.1f");
                ImGui::SameLine();
                if (ImGui::Button("reset G")){
                    imgui_color[1] = 0;
                }
                ImGui::SliderFloat("B",&imgui_color[2],-1.0f,1.0f,"%.1f");
                ImGui::SameLine();
                if (ImGui::Button("reset B")){
                    imgui_color[2] = 0;
                }
                ImGui::SliderFloat("A",&imgui_color[3],-1.0f,1.0f,"%.1f");
                ImGui::SameLine();
                if (ImGui::Button("reset A")){
                    imgui_color[3] = 0;
                }

                color.r = 255 * imgui_color[0];
                color.g = 255 * imgui_color[1];
                color.b = 255 * imgui_color[2];
                color.a = 255 * imgui_color[3];
            }

            void imGuiInfos(){
                ImGui::Text("Drawing %s",filename.c_str());
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