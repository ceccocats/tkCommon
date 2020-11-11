#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloudColorMaps.h"
#include "tkCommon/gui/shader/pointcloud4f.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4f : public Drawable {

        private:
            //Data
            int points;
            tk::data::CloudData*    cloud;
            tk::gui::Buffer<float>  glbuffer;

            //ColorMaps
            int selectedColorMap    = 0;
            std::string cloudcolor  = "Colored Cloud";
            std::vector<const char*> colorMaps;
            

            //Color values
            float min           = 999;
            float max           = -999;
            int axisShader      = -1;
            int useFeatureN     = 0;
            std::string text0   = "x axis";
            std::string text1   = "y axis";
            std::string text2   = "z axis";
            std::vector<const char*> features;

            //Monocolor pointcloud shader
            tk::gui::shader::pointcloud4f* monocolorCloud;
            
            void updateData(){
                float minv = 999;
                float maxv = -999;

                //Update feature viz list
                if(features.size() != (cloud->features.size()+3)){
                    features.clear();
                    features.push_back(text0.c_str());
                    features.push_back(text1.c_str());
                    features.push_back(text2.c_str());
                    for(auto const& f : cloud->features.keys())
                        features.push_back(f.c_str());
                }

                //Copy colud on gpu and features
                cloud->lock();
                points = cloud->points.cols();
                glbuffer.setData(cloud->points.data_h,cloud->points.size());
                if(useFeatureN > 2){
                    axisShader = -1;
                    //Copy feature               
                    tk::math::Vec<float> *f = &cloud->features[features[useFeatureN]];
                    glbuffer.setData(f->data_h, f->size(), cloud->points.size());

                    //Update min max
                    for(int i = 0; i < f->size(); i++){
                        float value = (*f)[i];
                        if(value > maxv) maxv = value;
                        if(value < minv) minv = value;
                    }
                }else{
                    axisShader = useFeatureN;
                    //Min Max axis
                    for(int i = 0; i < cloud->points.cols(); i++){
                        float value = cloud->points(useFeatureN,i);
                        if(value > maxv) maxv = value;
                        if(value < minv) minv = value;
                    }
                }
                cloud->unlockRead();

                if(min == 999){
                    min = minv;
                    max = maxv;
                }else{
                    min = 0.9*min + 0.1*minv;
                    max = 0.9*max + 0.1*maxv;
                }
            }

        public:

            //imgui settings
            tk::gui::Color_t color;
            float pointSize = 1.0f;        

            /**
             * @brief Construct a new Cloud 4f using one color
             * 
             * @param cloud pointcloud ref
             * @param color color
             */
            Cloud4f(tk::data::CloudData* cloud){
                this->points            = 0;
                this->cloud             = cloud;   
                this->selectedColorMap  = 0;
                this->color             = tk::gui::color::WHITE;
                this->useFeatureN       = 0;
                this->update            = true;
                this->tf                = tk::common::Tfpose::Identity();
            }

            ~Cloud4f(){

            }

            void updateRef(tk::data::CloudData* cloud){
                this->cloud = cloud;   
                update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::pointcloudColorMaps();
                glbuffer.init();

                tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;

                colorMaps.push_back(cloudcolor.c_str());
                for(int i = 0; i < shaderCloud->colormaps.size(); i++)
                    colorMaps.push_back(shaderCloud->colormaps[i].c_str());

                monocolorCloud = new tk::gui::shader::pointcloud4f();
            }

            void draw(tk::gui::Viewer *viewer){

                if(update == true){
                    min = 999;
                    max = -999;
                }

                if(cloud->isChanged() || update){
                    update = false;
                    updateData();
                }

                glPointSize(pointSize);
                glPushMatrix();
                glMultMatrixf(this->tf.matrix().data());
                if(selectedColorMap == 0){
                    monocolorCloud->draw(&glbuffer, points, color);
                }else{
                    tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
                    //std::cout<<shaderCloud->colormaps[selectedColorMap-1]<<",buffer,"<<points<<","<<min<<","<<max<<","<<axisShader<<","<<color.a()<<std::endl;
                    shaderCloud->draw(shaderCloud->colormaps[selectedColorMap-1], &glbuffer, points, min, max, axisShader, color.a());
                }
                glPopMatrix();
                glPointSize(1.0);		
            }

            void imGuiSettings(){
                ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
                ImGui::SliderFloat("Alpha",&color.a(),0,1.0f,"%.1f");
                if(ImGui::Combo("Color maps", &selectedColorMap, colorMaps.data(), colorMaps.size())){
                    update = true;
                }

                if(selectedColorMap == 0){
                    ImGui::ColorEdit3("Color", color.color);
                }else{
                    if(ImGui::Combo("feature", &useFeatureN, features.data(), features.size())){
                        update = true;
                    }
                }

            }

            void imGuiInfos(){
                std::stringstream print;
                print<<(*cloud);
                ImGui::Text("%s",print.str().c_str());
                print.clear();
            }

            void onClose(){
                tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
                shaderCloud->close();
                glbuffer.release();
                delete shader;

                monocolorCloud->close();
                delete monocolorCloud;
            }

            std::string toString(){
                return cloud->header.name;
            }
	};
}}