#pragma once

#include "tkCommon/data/gen/SonarData_gen.h"
#include "tkCommon/data/CloudData.h"

namespace tk{ namespace data{

    class SonarData : public SonarData_gen{
        public:

        void init(int w, int h, float resolution){
            SonarData_gen::init();
            resize(w,h,resolution);
        }

        void resize(int w, int h, float resolution){
            this->image.init(w,h,1);
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