#pragma once

#include "tkCommon/data/gen/SonarData_gen.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace data{

    class SonarData : public SonarData_gen{
        public:

        void init(int w, int h_i, int h_r, float resolution){
            SonarData_gen::init();
            resize(w,h_i,h_r,resolution);
        }

        void resize(int w, int h_i, int h_r, float resolution){
            this->raw.init(w,h_r,1);
            this->image.init(w,h_i,1);
            this->azimuth.resize(w,1);
            this->resolution = resolution;
        }

        SonarData& operator=(const SonarData& s) {
            SonarData_gen::operator=(s);
            return *this;
        }

        void toKitty(const std::string& fileName){
            image.toKitty(fileName);
        }
        
        void fromKitty(const std::string& fileName){
            image.fromKitty(fileName);
        }


#ifdef ROS_ENABLED
        void toRos(sensor_msgs::Image &msg) {
            image.toRos(msg);        
        }

        void fromRos(sensor_msgs::Image &msg) {
            image.fromRos(msg);
        }
#endif

    };
}}