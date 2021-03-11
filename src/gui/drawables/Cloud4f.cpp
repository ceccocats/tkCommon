#include "tkCommon/gui/drawables/Cloud4f.h"

void 
tk::gui::Cloud4f::updateData(){

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
                glbuffer.setData(f->data(), f->size(), offset);
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
                        resetMinMax   = false;
                        minMax[0][ch] = min;
                        minMax[1][ch] = max;
                    }else{
                        minMax[0][ch] = 0.95*minMax[0][ch] + 0.05*min;
                        minMax[1][ch] = 0.95*minMax[1][ch] + 0.05*max;
                    }
                }else{
                    minMax[0][ch] = 0.0f;
                    minMax[1][ch] = 1.0f;
                }
            }
        }
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
                        resetMinMax  = false;
                        minMax[0][0] = min;
                        minMax[1][0] = max;
                    }else{
                        minMax[0][0] = 0.95*minMax[0][0] + 0.05*min;
                        minMax[1][0] = 0.95*minMax[1][0] + 0.05*max;
                    }
                }else{
                    minMax[0][0] = 0.0f;
                    minMax[1][0] = 1.0f;
                }
            }
        }

        //using features
        if(selected[0] > 2){
            axisShader = -1;
            tk::math::Vec<float> *f = &cloud->features[features[selected[0]]];
            tkASSERT(f->size() == points,"Cloud corrupted\n");
            glbuffer.setData(f->data(), f->size(), cloud->points.size());

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
            }else{
                    minMax[0][0] = 0.0f;
                    minMax[1][0] = 1.0f;
                }
        }
        return;
    }
}

tk::gui::Cloud4f::Cloud4f(std::string name){
    //CloudParams
    this->name         = name;
    this->points       =  0;
    this->color        =  tk::gui::color::WHITE;
    this->color.a()    =  0.5;
    this->resetMinMax  =  true;

    //DefaultSelected
    this->cloudMod          = 0;
    this->selectedColorMap  = cloudMod0.second;
    this->selected[0]       = feature0.second;
    this->selected[1]       = featuresChannel0.second;
    this->selected[2]       = featuresChannel0.second;
    this->selected[3]       = featuresChannel0.second;
}

tk::gui::Cloud4f::Cloud4f(tk::data::CloudData* cloud, std::string name) : Cloud4f(name){
    this->cloud =  cloud;
}

tk::gui::Cloud4f::~Cloud4f(){

}

void 
tk::gui::Cloud4f::onInit(tk::gui::Viewer *viewer){
    
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


void 
tk::gui::Cloud4f::updateData(int i, tk::gui::Viewer *viewer){

    cloud = (tk::data::CloudData*)data[0];
    points = cloud->points.cols();
    this->tf = cloud->header.tf;
    glbuffer.setData(cloud->points.data(),cloud->points.size());
    print.str("");
    print<<(*cloud);
    updateData();
}

void 
tk::gui::Cloud4f::drawData(tk::gui::Viewer *viewer){
    
    glPointSize(pointSize);
    if(cloudMod == cloudMod0.second){
        monocolorCloud->draw(drwModelView,&glbuffer, points, color);
    }
    if(cloudMod == cloudMod1.second){
        pointcloudrgba->draw(drwModelView,&glbuffer,points,
            selected[1]>0,minMax[0][1],minMax[1][1],
            selected[2]>0,minMax[0][2],minMax[1][2],
            selected[3]>0,minMax[0][3],minMax[1][3],
            color.a());
    }
    if(cloudMod == cloudMod2.second){
        tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
        shaderCloud->draw(drwModelView,shaderCloud->colormaps[selectedColorMap], &glbuffer, 
            points, minMax[0][0], minMax[1][0], axisShader, color.a());
    }
    glPointSize(1.0);		
}

void 
tk::gui::Cloud4f::imGuiSettings(){
    ImGui::SliderFloat("Size",&pointSize,1.0f,20.0f,"%.1f");
    ImGui::SliderFloat("Alpha",&color.a(),0,1.0f,"%.2f");
    if(ImGui::Combo("Draw mode", &cloudMod, cloudMods.data(), cloudMods.size())){
        resetMinMax = true;
    }
    ImGui::Separator();

    //Color cloud
    if(cloudMod == cloudMod0.second){
        ImGui::ColorEdit3("Color", color.color);
    }

    //RGBA cloud
    if(cloudMod == cloudMod1.second){
        if(ImGui::Combo("feature r", &selected[1], featuresChannels.data(), featuresChannels.size())){
            resetMinMax = true;
            forceUpdate();
        }
        if(selected[1] > 0){
            if(ImGui::Combo("feature g", &selected[2], featuresChannels.data(), featuresChannels.size())){
                resetMinMax = true;
                forceUpdate();
            }
            if(selected[2] > 0){
                if(ImGui::Combo("feature b", &selected[3], featuresChannels.data(), featuresChannels.size())){
                    resetMinMax = true;
                    forceUpdate();
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
            resetMinMax = true;
            forceUpdate();
        }
        ImGui::Text("Min %f Max %f", minMax[0][0], minMax[1][0]);
        return;
    }
}

void 
tk::gui::Cloud4f::imGuiInfos(){
    ImGui::Text("%s",print.str().c_str());
}

void 
tk::gui::Cloud4f::onClose(){
    tk::gui::shader::pointcloudColorMaps* shaderCloud = (tk::gui::shader::pointcloudColorMaps*) shader;
    shaderCloud->close();
    glbuffer.release();
    delete shaderCloud;

    monocolorCloud->close();
    delete monocolorCloud;

    pointcloudrgba->close();
    delete pointcloudrgba;
}