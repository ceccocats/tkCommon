#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloudColorMaps.h"
#include "tkCommon/gui/shader/pointcloudRGBA.h"
#include "tkCommon/gui/shader/pointcloud4f.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4f : public Drawable {

        public:
            tk::data::CloudData*    cloud;

        private:
            int points;
            tk::gui::Buffer<float>  glbuffer;


            std::pair<std::string,int>  cloudMod0 = {"Colored Cloud",0};
            std::pair<std::string,int>  cloudMod1 = {"RGBA Cloud",1};
            std::pair<std::string,int>  cloudMod2 = {"Maps Cloud",2};
            std::vector<const char*>    cloudMods;

            std::pair<std::string,int>  feature0 = {"axis x",0};
            std::pair<std::string,int>  feature1 = {"axis y",1};
            std::pair<std::string,int>  feature2 = {"axis z",2};
            std::vector<const char*>    features;

            std::pair<std::string,int>  featuresChannel0 = {"---",0};
            std::vector<const char*>    featuresChannels;

            std::vector<const char*>    colorMaps;

            tk::gui::shader::pointcloud4f*   monocolorCloud;
            tk::gui::shader::pointcloudRGBA* pointcloudrgba;

            bool resetMinMax;
            
            void updateData(){

                //Colored Cloud
                if(cloudMod == cloudMod0.second){
                    return;
                }

                //RGBA Cloud
                if(cloudMod == cloudMod1.second){
                    //Update viz menu
                    if(featuresChannels.size() != (cloud->features.size()+1)){
                        featuresChannels.clear();
                        featuresChannels.push_back(featuresChannel0.first.c_str());
                        for(auto const& f : cloud->features.keys())
                            featuresChannels.push_back(f.c_str());
                    }

                    //channels
                    int offset = cloud->points.size();
                    for(int ch = 1; ch < 4; ch++){
                        if(selected[ch] != featuresChannel0.second){
                            tk::math::Vec<float> *f = &cloud->features[featuresChannels[selected[ch]]];
                            tkASSERT(f->size() == points,"Cloud corrupted\n");
                            glbuffer.setData(f->data_h, f->size(), offset);
                            offset += f->size();
                            if(autoMinMax == true){
                                float min =  999;
                                float max = -999;
                                for(int i = 0; i < f->size(); i++){
                                    float value = (*f)[i];
                                    if(value > max) max = value;
                                    if(value < min) min = value;
                                }
                                if(resetMinMax == true){
                                    minMax[0][ch] = min;
                                    minMax[1][ch] = max;
                                }else{
                                    minMax[0][ch] = 0.95*minMax[0][ch] + 0.05*min;
                                    minMax[1][ch] = 0.95*minMax[1][ch] + 0.05*max;
                                }
                            }
                        }
                    }
                    resetMinMax = false;
                    return;
                }


                //Feature cloud
                if(cloudMod == cloudMod2.second){
                    //Update feature viz list
                    if(features.size() != (cloud->features.size()+3)){
                        features.clear();
                        features.push_back(feature0.first.c_str());
                        features.push_back(feature1.first.c_str());
                        features.push_back(feature2.first.c_str());
                        for(auto const& f : cloud->features.keys())
                            features.push_back(f.c_str());
                    }

                    //using axis like feature
                    for(int axis = 0; axis < 3; axis++){
                        if(selected[0] == axis){
                            axisShader = axis;
                            if(autoMinMax == true){
                                float min =  999;
                                float max = -999;
                                for(int i = 0; i < cloud->points.cols(); i++){
                                    float value = cloud->points(axis,i);
                                    if(value > max) max = value;
                                    if(value < min) min = value;
                                }
                                if(resetMinMax == true){
                                    resetMinMax   = false;
                                    minMax[0][0] = min;
                                    minMax[1][0] = max;
                                }else{
                                    minMax[0][0] = 0.95*minMax[0][0] + 0.05*min;
                                    minMax[1][0] = 0.95*minMax[1][0] + 0.05*max;
                                }
                            }
                        }
                    }

                    //using features
                    if(selected[0] > 2){
                        axisShader = -1;
                        tk::math::Vec<float> *f = &cloud->features[features[selected[0]]];
                        tkASSERT(f->size() == points,"Cloud corrupted\n");
                        glbuffer.setData(f->data_h, f->size(), cloud->points.size());
                        if(autoMinMax == true){
                            float min =  999;
                            float max = -999;
                            for(int i = 0; i < f->size(); i++){
                                float value = (*f)[i];
                                if(value > max) max = value;
                                if(value < min) min = value;
                            }
                            if(resetMinMax == true){
                                resetMinMax  = false;
                                minMax[0][0] = min;
                                minMax[1][0] = max;
                            }else{
                                minMax[0][0] = 0.95*minMax[0][0] + 0.05*min;
                                minMax[1][0] = 0.95*minMax[1][0] + 0.05*max;
                            }
                        }
                    }
                    return;
                }
            }

        public:

            /**
             * @brief 
             * Color used in monocolorCloud 
             */
            tk::gui::Color_t color;
            /**
             * @brief 
             * Auto min max calculation
             */
            bool  autoMinMax = true;
             /**
             * @brief 
             * Pointcloud size (pixel)
             */
            float pointSize = 1.0f;   
            /**
             * @brief 
             * Selected cloud drawing
             */     
            int   cloudMod;
            /**
             * @brief 
             * Selected color map
             */
            int   selectedColorMap;
            /**
             * @brief 
             * wich axis to use like feature 
             */
            int axisShader = -1;
            /** 
             * @brief
             * 0:Feature, 1:FeatureR, 2:FeatureG, 3:FeatureB 
             */
            int   selected[4];
            /**
             * @brief 
             * c0:Feature, c1:FeatureR, c2:FeatureG, c3:FeatureB
             * r0: min, r1:max
             */
            float minMax[2][4];
            /**
             * @brief 
             * Method that update min and max 
             */
            bool updateMinMax;
            
            /**
             * @brief Construct a new Cloud 4f using one color
             * 
             * @param cloud pointcloud ref
             * @param color color
             */
            Cloud4f(tk::data::CloudData* cloud){
                //CloudParams
                this->points       =  0;
                this->cloud        =  cloud; 
                this->color        =  tk::gui::color::WHITE;
                this->color.a()    =  0.5;
                this->update       =  true;
                this->resetMinMax  =  true;
                this->updateMinMax =  false;

                //DefaultSelected
                this->cloudMod          = 0;
                this->selectedColorMap  = cloudMod0.second;
                this->selected[0]       = feature0.second;
                this->selected[1]       = featuresChannel0.second;
                this->selected[2]       = featuresChannel0.second;
                this->selected[3]       = featuresChannel0.second;
            }

            ~Cloud4f(){

            }

            void updateRef(tk::data::CloudData* cloud){
                this->cloud = cloud;   
                update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                
                //init shaders
                monocolorCloud  = new tk::gui::shader::pointcloud4f();
                pointcloudrgba  = new tk::gui::shader::pointcloudRGBA();
                shader          = new tk::gui::shader::pointcloudColorMaps();
                glbuffer.init();

                //fill data for menu
                tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
                for(int i = 0; i < shaderCloud->colormaps.size(); i++)
                    colorMaps.push_back(shaderCloud->colormaps[i].c_str());

                //fill data for colormaps
                cloudMods.push_back(cloudMod0.first.c_str());
                cloudMods.push_back(cloudMod1.first.c_str());
                cloudMods.push_back(cloudMod2.first.c_str());
            }

            void draw(tk::gui::Viewer *viewer){

                if(updateMinMax == true){
                    updateMinMax = false;
                    resetMinMax  = true;
                    cloud->lock();
                    if(points != cloud->points.cols()){
                        points = cloud->points.cols();
                        glbuffer.setData(cloud->points.data_h,cloud->points.size());
                        updateData();
                    }else{
                        updateData();
                    }
                    cloud->unlockRead();
                }

                if(cloud->isChanged() || update){
                    update = false;
                    cloud->lock();
                    points = cloud->points.cols();
                    glbuffer.setData(cloud->points.data_h,cloud->points.size());
                    updateData();
                    cloud->unlockRead();
                }

                glPointSize(pointSize);
                glPushMatrix();
                glMultMatrixf(this->tf.matrix().data());
                if(cloudMod == cloudMod0.second){
                    monocolorCloud->draw(&glbuffer, points, color);
                }
                if(cloudMod == cloudMod1.second){
                    pointcloudrgba->draw(&glbuffer,points,
                        selected[1]>0,minMax[0][1],minMax[1][1],
                        selected[2]>0,minMax[0][2],minMax[1][2],
                        selected[3]>0,minMax[0][3],minMax[1][3],
                        color.a());
                }
                if(cloudMod == cloudMod2.second){
                    tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
                    shaderCloud->draw(shaderCloud->colormaps[selectedColorMap], &glbuffer, 
                        points, minMax[0][0], minMax[1][0], axisShader, color.a());
                }
                glPopMatrix();
                glPointSize(1.0);		
            }

            void imGuiSettings(){
                ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
                ImGui::SliderFloat("Alpha",&color.a(),0,1.0f,"%.2f");
                if(ImGui::Combo("Draw mode", &cloudMod, cloudMods.data(), cloudMods.size())){
                    updateMinMax = true;
                }
                ImGui::Separator();

                //Color cloud
                if(cloudMod == cloudMod0.second){
                    ImGui::ColorEdit3("Color", color.color);
                    return;
                }

                //RGBA cloud
                if(cloudMod == cloudMod1.second){
                    if(ImGui::Combo("feature r", &selected[1], featuresChannels.data(), featuresChannels.size())){
                        updateMinMax = true;
                    }
                    if(selected[1] > 0){
                        if(ImGui::Combo("feature g", &selected[2], featuresChannels.data(), featuresChannels.size())){
                            updateMinMax = true;
                        }
                        if(selected[2] > 0){
                            if(ImGui::Combo("feature b", &selected[3], featuresChannels.data(), featuresChannels.size())){
                                updateMinMax = true;
                            }
                        }else{
                            selected[3] = 0;
                        }
                    }else{
                        selected[2] = 0;
                        selected[3] = 0;
                    }
                    return;
                }

                //Feature cloud
                if(cloudMod == cloudMod2.second){
                    ImGui::Combo("Color maps", &selectedColorMap, colorMaps.data(), colorMaps.size());
                    if(ImGui::Combo("feature", &selected[0], features.data(), features.size())){
                        updateMinMax = true;
                    }
                    ImGui::Text("Min %f Max %f", minMax[0][0], minMax[1][0]);
                    return;
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
                delete shaderCloud;

                monocolorCloud->close();
                delete monocolorCloud;

                pointcloudrgba->close();
                delete pointcloudrgba;
            }

            std::string toString(){
                return cloud->header.name;
            }
	};
}}