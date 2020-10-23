#pragma once
#include "tkCommon/gui/Drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloud4fFeatures.h"
#include "tkCommon/gui/shader/pointcloud4f.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4f : public Drawable {

        private:
            tk::data::CloudData*    cloud;
            tk::gui::Buffer<float>  glbuffer;
            float                   pointSize = 1.0f;
            bool                    update;

            static const int    maxItems = 30;
            int                 nItems;
            const char*         items[maxItems];
            int                 selectedColorMap = 0;

            int         useFeatureN;
            int         nFeatures;
            const char* useFeatures[maxItems];

            std::string cloudcolor = "Colored Cloud";
            tk::gui::shader::pointcloud4f* shCloudColor;

            std::string axisx = "axis x";
            std::string axisy = "axis y";
            std::string axisz = "axis z";

            float  min, max;
            void minMaxFeatures(){
                min = 99999;
                max = -99999;

                // use x, y or z
                if(useFeatureN < 3){
                    for(int i = 0; i < cloud->points.cols(); i++){
                         float value = cloud->points(useFeatureN,i);

                        if(value > max) max = value;
                        if(value < min) min = value;
                    }
                
                //use feature ONLY I NOWWWWW
                }else{
                    const tk::math::Vec<float> *f = &cloud->features[tk::data::CloudData::FEATURES_I];

                    for(int i = 0; i < f->size(); i++){
                        float value = f->data_h[i];/////////////////////For now I

                        if(value > max) max = value;
                        if(value < min) min = value;
                    }
                }
            }

        public:
            tk::gui::Color_t        color;

            /**
             * @brief Construct a new Cloud 4f using one color
             * 
             * @param cloud pointcloud ref
             * @param color color
             */
            Cloud4f(tk::data::CloudData* cloud){
                this->cloud             = cloud;   
                this->selectedColorMap  = 0;
                this->color             = tk::gui::color::WHITE;
                this->useFeatureN       = 0;
                this->update            = true;
            }

            /**
             * @brief Construct a new Cloud 4f object drawing colormap using one axis
             * 
             * @param cloud             pointcloud ref
             * @param selectedColorMap  colormap to use
             * @param useAxis           axis to use
             */
            Cloud4f(tk::data::CloudData* cloud, int selectedColorMap, int useAxis){
                this->cloud             = cloud;   
                this->selectedColorMap  = selectedColorMap;
                this->color             = tk::gui::color::WHITE;
                this->useFeatureN       = useAxis > 3 || useAxis < 0 ? 0 : useAxis;
            }

            /**
             * @brief Construct a new Cloud 4f object drawing colormap using a feature
             * 
             * @param cloud             pointcloud ref
             * @param selectedColorMap  colormap to use
             * @param feature           feature to use
             */
            Cloud4f(tk::data::CloudData* cloud, int selectedColorMap, tk::data::CloudData_gen::featureType_t feature){
                this->cloud             = cloud;   
                this->selectedColorMap  = selectedColorMap;
                this->color             = tk::gui::color::WHITE;
                this->useFeatureN       = 0; //TODO cloud->features_map[feature] + 3;
            }

            ~Cloud4f(){

            }

            void updateRef(tk::data::CloudData* cloud){
                this->cloud = cloud;   
                update = true;
            }

            void onInit(tk::gui::Viewer *viewer){
                shader = new tk::gui::shader::pointcloud4fFeatures();
                glbuffer.init();

                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;

                items[0] = cloudcolor.c_str();
                nItems = 1;
                for(int i = 0; i < shaderCloud->colormaps.size(); i++){
                    if(maxItems < i+1){
                        clsWrn("too many colormaps, change maxItems. Skipping\n")
                    }else{
                        items[i+1] = shaderCloud->colormaps[i].c_str();
                        nItems = i+2;
                    }
                }

                shCloudColor = new tk::gui::shader::pointcloud4f();

                useFeatures[0] = axisx.c_str();
                useFeatures[1] = axisy.c_str();
                useFeatures[2] = axisz.c_str();
                nFeatures = 3;
            }

            void draw(tk::gui::Viewer *viewer){
               if(cloud->isChanged() || update){
                   update = false;

                    int i = 3;
                    for (auto const& f : cloud->features.keys()){
                        useFeatures[i] = f.c_str();
                        i++;
                        nFeatures = i;
                    }

                    //const tk::math::Vec<float> *f = &cloud->features[tk::data::CloudData::FEATURES_I];

                    cloud->lock();
                    glbuffer.setData(cloud->points.data_h,cloud->points.size());
                    /*if(selectedColorMap != 0 && useFeatureN > 2){
                        glbuffer.setData(f->data_h, f->size(), cloud->points.size());
                    }*/
                    cloud->unlockRead();
                    minMaxFeatures();
                }

                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;
                glPointSize(pointSize);
                if(selectedColorMap == 0){
                    shCloudColor->draw(&glbuffer, cloud->points.size()/4,color);
                }else{
                    if(useFeatureN > 2){
                        //shaderCloud->draw(shaderCloud->colormaps[selectedColorMap-1],&glbuffer, glbuffer.size()/5,min,max,0);
                    }else{
                        if(useFeatureN == 0) //Drawing x
                            shaderCloud->draw(shaderCloud->colormaps[selectedColorMap-1],&glbuffer, cloud->points.cols(),min,max,1);
                        if(useFeatureN == 1) //Drawing y
                            shaderCloud->draw(shaderCloud->colormaps[selectedColorMap-1],&glbuffer, cloud->points.cols(),min,max,2);
                        if(useFeatureN == 2) //Drawing z
                            shaderCloud->draw(shaderCloud->colormaps[selectedColorMap-1],&glbuffer, cloud->points.cols(),min,max,3);
                    }
                    
                }
                glPointSize(1.0);		
            }

            void imGuiSettings(){
                ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
                if(ImGui::Combo("Color maps", &selectedColorMap, items, nItems)){
                    update = true;
                }

                if(selectedColorMap == 0){
                    ImGui::ColorEdit4("Color", color.color);
                }else{
                    if(ImGui::Combo("feature", &useFeatureN, useFeatures, nFeatures)){
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
                tk::gui::shader::pointcloud4fFeatures* shaderCloud = (tk::gui::shader::pointcloud4fFeatures*) shader;
                shaderCloud->close();
                glbuffer.release();
                delete shader;

                shCloudColor->close();
                delete shCloudColor;
            }

            std::string toString(){
                return cloud->header.name;
            }
	};
}}