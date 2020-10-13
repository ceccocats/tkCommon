#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloud4fFeatures.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4fFeatures : public Drawable {

        private:
            tk::data::CloudData*    cloud;
            tk::gui::Buffer<float>  glbuffer;
            float                   pointSize = 1.0f;

            static const int    maxItems = 30;
            const char*         items[maxItems];
            int                 selectedColorMap = 0;

            float  min, max;

            void minMaxFeatures(){
                min = 99999;
                max = -99999;

                for(int i = 0; i < cloud->features.cols(); i++){
                    float value = cloud->features.data_h[i];

                    if(value > max) max = value;
                    if(value < min) min = value;
                }
            }

        public:

            Cloud4fFeatures(tk::data::CloudData* cloud, int selectedColorMap = 0){
                this->cloud = cloud;   
                this->selectedColorMap = selectedColorMap;
            }

            ~Cloud4fFeatures(){

            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::pointcloud4fFeatures();
                glbuffer.init();

                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;

                for(int i = 0; i < shaderCloud->colormaps.size(); i++){
                    if(maxItems < i){
                        clsWrn("too many colormaps, change maxItems. Skipping\n")
                    }else{
                        items[i] = shaderCloud->colormaps[i].c_str();
                    }
                }
            }

            void draw(tk::gui::Viewer *viewer){
               if(cloud->isChanged()){
                   //Controllo fino a che non si definisce
                   tkASSERT(cloud->features.cols() == cloud->features.size() && cloud->features.size() > 0, 
                        "controllo temporaneo fino alla definizione della matrice features")

                    cloud->lock();
                    glbuffer.setData(cloud->points.data_h,cloud->points.size());
                    glbuffer.setData(cloud->features.data_h,cloud->features.cols(),cloud->points.size());
                    cloud->unlockRead();
                    minMaxFeatures();
                }

                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;
                glPointSize(pointSize);
                shaderCloud->draw(shaderCloud->colormaps[selectedColorMap],&glbuffer, glbuffer.size()/5,min,max);
                glPointSize(1.0);		
            }

            void imGuiSettings(){
                ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
                ImGui::Combo("combo", &selectedColorMap, items, sizeof(const char *));
            }

            void imGuiInfos(){
                ImGui::Text("Pointcloud has %d points",glbuffer.size());
            }

            void onClose(){
                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;
                shaderCloud->close();
                glbuffer.release();
                delete shader;
            }

            std::string toString(){
                return cloud->header.name;
            }
	};
}}