#pragma once
#include "tkCommon/gui/drawables/Drawable.h"
#include "tkCommon/gui/shader/pointcloudColorMaps.h"
#include "tkCommon/gui/shader/pointcloudRGBA.h"
#include "tkCommon/gui/shader/pointcloud4f.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace gui{

	class Cloud4f : public Drawable {

        private:
            tk::data::CloudData*    cloud       = nullptr;
            tk::data::CloudData*    cldUpdate   = nullptr;

            uint32_t counter    = 0;
            bool updateCld      = false;

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

            std::string name;

            std::stringstream print;
            
            void updateData();
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
            Cloud4f(std::string name = "cloud4f");
            Cloud4f(tk::data::CloudData* cloud, std::string name = "cloud4f");
            ~Cloud4f();

            void updateRef(tk::data::CloudData* cloud);
            void onInit(tk::gui::Viewer *viewer);
            void draw(tk::gui::Viewer *viewer);
            void imGuiSettings();
            void imGuiInfos();
            void onClose();
            
            std::string toString();
	};
}}