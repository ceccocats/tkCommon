#pragma once
#include "tkCommon/data/gen/CloudData_gen.h"

#ifdef TKROS_ENABLED

#if TKROS_VERSION == 1
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/point_cloud_conversion.h>
#endif
#if TKROS_VERSION == 2
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <sensor_msgs/msg/laser_scan.hpp>
#include <sensor_msgs/point_cloud_conversion.hpp>
#endif

#endif

#ifdef PCL_ENABLED
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

#ifdef PCL_ENABLED
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

namespace tk { namespace data {

    class CloudData : public CloudData_gen{

        private:
            double lo_state = -1.0;
            double hi_state = -1.0;
        
        public:

        void print() {
            points.print();
            const auto &k = features.keys();
            const auto &f = features.vals();
            for(int i=0; i<f.size(); i++) {
                std::cout<<k[i]<<":\n";
                f[i]->print();
            }
        }

        void resize(int n) {
            points.resize(4, n);
            ranges.resize(3, n);
            const auto &fts = features.vals();
            for(int i=0; i<fts.size(); i++) {
                fts[i]->resize(n);
            }
            tkASSERT(checkConsistency());
        }

        int size() {
            return points.cols();
        }

        bool checkConsistency() {
            int n = size();
            if(n>0 && points.rows() != 4) return false;
            if(n>0 && ranges.rows() != 3) return false;
            if(n != points.cols()) return false;
            if(n != ranges.cols()) return false;
            for(int i=0; i<features.keys().size(); i++) {
                if(features[features.keys()[i]].size() != n)
                    return false;
            }
            return true;
        }

        /**
         *  WARING: slow do not use in performance loop
         */
        void setPoint(int pos, tk::math::Vec3f pt) {
            points(0, pos) = pt.x();
            points(1, pos) = pt.y();
            points(2, pos) = pt.z();
            points(3, pos) = 1;
        }
        /**
         *  WARING: slow do not use in performance loop
         */
        void setFeature(int pos, featureType_t f, float val) {
            features[f][pos] = val;
        }

        void addFeature(featureType_t f) {
            features.add(f);
            features[f].resize(size());
        }

        void bounds(float &aMinX, float &aMaxX, float &aMinY, float &aMaxY, float &aMinZ, float &aMaxZ) {
            aMinX = points.matrix().row(0).minCoeff();
            aMaxX = points.matrix().row(0).maxCoeff();

            aMinY = points.matrix().row(1).minCoeff();
            aMaxY = points.matrix().row(1).maxCoeff(); 

            aMinZ = points.matrix().row(2).minCoeff();
            aMaxZ = points.matrix().row(2).maxCoeff();
        }

        void gammaCorrectionIntensity(){
            //From ouster

            if(features.size() == 0){
                tkERR("Empty\n");
                    return;
            }

            if(!features.exists(tk::data::CloudData::FEATURES_I)){
                    tkERR("Error\n");
                    return;
            }
            tk::math::Vec<float> *intensity = &features[tk::data::CloudData::FEATURES_I];

            const size_t n = intensity->size();
            const size_t kth_extreme = n / 100;
            std::vector<size_t> indices(n);
            for (size_t i = 0; i < n; i++) {
                indices[i] = i;
            }
            auto cmp = [&](const size_t a, const size_t b) {
                return intensity->data()[a] < intensity->data()[b];
            };
            std::nth_element(indices.begin(), indices.begin() + kth_extreme,
                                indices.end(), cmp);
            const double lo = intensity->data()[*(indices.begin() + kth_extreme)];
            std::nth_element(indices.begin() + kth_extreme,
                                indices.end() - kth_extreme, indices.end(), cmp);
            const double hi = intensity->data()[*(indices.end() - kth_extreme)];
            if (lo_state < 0) {
                lo_state = lo;
                hi_state = hi;
            }
            lo_state = 0.9 * lo_state + 0.1 * lo;
            hi_state = 0.9 * hi_state + 0.1 * hi;
            
            // Apply gamma correction
            for(int i = 0; i < intensity->size(); i++){

                double value;
                value = (intensity->data()[i] - lo) / (hi - lo);
                value = sqrt(value);
                if(value > 1.0f) value = 1.0f;
                if(value < 0.0f) value = 0.0f;
                intensity->operator[](i) = float(value);
            }
        }

        void toKitty(const std::string& fileName){
            std::ofstream output(fileName.c_str(), std::ios::out | std::ios::binary);
            if(!output.good()){
                tkERR("Error creating file "<<fileName<<"\n");
            }   

            bool hasF = false;
            if (features.exists(tk::data::CloudData_gen::FEATURES_I)) {
                hasF = true;
            }  

            float data[4];
            for(int i = 0; i < points.cols(); i++){
                data[0] = points(0,i);
                data[1] = points(1,i);
                data[2] = points(2,i);
                if(hasF){
                    auto intensity = &features[tk::data::CloudData_gen::FEATURES_I];
                    data[3] = (*intensity)[i];
                }else{
                    data[3] = 0;
                }
                output.write((char *) data, 4*sizeof(float));
            }
            output.close();
        }

        void fromKitty(const std::string& fileName){
            std::fstream input(fileName.c_str(), std::ios::in | std::ios::binary);
            if(!input.good()){
                tkERR("Error loading file\n");
            }
            input.seekg(0, std::ios::beg);           

            std::vector<tk::common::Vector4<float>> data;            
            for (int i = 0; input.good() && !input.eof(); i++) {
                tk::common::Vector4<float> point;
                input.read((char *) &point.x(), sizeof(float));
                input.read((char *) &point.y(), sizeof(float));
                input.read((char *) &point.z(), sizeof(float));
                input.read((char *) &point.w(), sizeof(float));
                data.push_back(point);
            }
            input.close();

            if (!features.exists(tk::data::CloudData_gen::FEATURES_I)) {
                features.add(tk::data::CloudData_gen::FEATURES_I);
            } 
            resize(data.size());

            auto intensity = &features[tk::data::CloudData_gen::FEATURES_I];
            for(int i = 0; i < data.size(); i++){
                points(0,i) = data[i].x();
                points(1,i) = data[i].y();
                points(2,i) = data[i].z();
                points(3,i) = 1;
                (*intensity)[i] = data[i].w();
            }
        }

#ifdef TKROS_ENABLED

#if TKROS_VERSION == 1
        void toRos(sensor_msgs::LaserScan &msg) {
#endif
#if TKROS_VERSION == 2
        void toRos(sensor_msgs::msg::LaserScan &msg) {
#endif
            this->header.toRos(msg.header);
            tkWRN("Not implemented.\n");
        }

#if TKROS_VERSION == 1
        void toRos(sensor_msgs::PointCloud2 &msg) {
#endif
#if TKROS_VERSION == 2
        void toRos(sensor_msgs::msg::PointCloud2 &msg) {
#endif
            this->header.toRos(msg.header);
            msg.width  = size();
            msg.height = 1;
            msg.is_bigendian = false;
            msg.is_dense = true;

            //fields setup
            sensor_msgs::PointCloud2Modifier modifier(msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");

            bool hasI, hasC, hasT;
            hasI = hasC = hasT = false;
            if (features.exists(tk::data::CloudData_gen::FEATURES_I)) {
#if TKROS_VERSION == 1
                msg.point_step = addPointField(msg, "intensity", 1, sensor_msgs::PointField::FLOAT32, msg.point_step);
#endif
#if TKROS_VERSION == 2
                msg.point_step = addPointField(msg, "intensity", 1, sensor_msgs::msg::PointField::FLOAT32, msg.point_step);
#endif          
                hasI = true;
            }
            if (features.exists(tk::data::CloudData_gen::FEATURES_CHANNEL)) {
#if TKROS_VERSION == 1
                msg.point_step = addPointField(msg, "ring", 1, sensor_msgs::PointField::UINT16, msg.point_step);
#endif
#if TKROS_VERSION == 2
                msg.point_step = addPointField(msg, "ring",      1, sensor_msgs::msg::PointField::UINT16,  msg.point_step);
#endif
                hasC = true;
            }
            if (features.exists(tk::data::CloudData_gen::FEATURES_TIME)) {

#if TKROS_VERSION == 1
                msg.point_step = addPointField(msg, "time",      1, sensor_msgs::PointField::FLOAT32, msg.point_step);
#endif
#if TKROS_VERSION == 2
                msg.point_step = addPointField(msg, "time",      1, sensor_msgs::msg::PointField::FLOAT32, msg.point_step);
#endif
                hasT = true;
            }
            msg.row_step = msg.width * msg.point_step;
            msg.data.resize(msg.height * msg.row_step);
            
            sensor_msgs::PointCloud2Iterator<float>     iter_x(msg, "x");
            sensor_msgs::PointCloud2Iterator<float>     iter_y(msg, "y");
            sensor_msgs::PointCloud2Iterator<float>     iter_z(msg, "z");
            sensor_msgs::PointCloud2Iterator<float>     *iter_intensity, *iter_time;
            sensor_msgs::PointCloud2Iterator<uint16_t>  *iter_ring;
            tk::math::Vec<float>                        *fI, *fC, *fT;
            if (hasI) {
                iter_intensity  = new sensor_msgs::PointCloud2Iterator<float>(msg, "intensity");
                fI              = &features[tk::data::CloudData_gen::FEATURES_I];
            }
            if (hasC) {
                iter_ring       = new sensor_msgs::PointCloud2Iterator<uint16_t>(msg, "ring");
                fC              = &features[tk::data::CloudData_gen::FEATURES_CHANNEL];
            }
            if (hasT) {
                iter_time       = new sensor_msgs::PointCloud2Iterator<float>(msg, "time");
                fT              = &features[tk::data::CloudData_gen::FEATURES_TIME];
            }

            for(int i=0; i<size(); i++) {
                *iter_x         = points(0,i);     ++iter_x; 
                *iter_y         = points(1,i);     ++iter_y; 
                *iter_z         = points(2,i);     ++iter_z; 
                if (hasI) {
                    **iter_intensity = (*fI)[i]; 
                    ++(*iter_intensity); 
                }
                if (hasC) {
                    **iter_ring      = (*fC)[i]; 
                    ++(*iter_ring);
                }   
                if (hasT) {
                    **iter_time      = (*fT)[i]; 
                    ++(*iter_time);  
                }
            }

        }

#if TKROS_VERSION == 1
        void fromRos(sensor_msgs::LaserScan &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(sensor_msgs::msg::LaserScan &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::CLOUD; 
            
            tk::math::Vec<float>* intensity = nullptr;
            if (this->features.exists(FEATURES_I))
                tk::math::Vec<float>* intensity = &features[FEATURES_I];

            // fill points
            resize(msg.ranges.size());
            for(int i = 0 ; i < msg.ranges.size(); i++) {
                float r = msg.ranges[i];
                if (msg.ranges[i] < msg.range_min || msg.ranges[i] > msg.range_max)
                    r = 0;

                float angle = msg.angle_min + msg.angle_increment * float(i);

                this->points(0, i) = r * cos(angle);
                this->points(1, i) = r * sin(angle);
                this->points(2, i) = 0;
                this->points(3, i) = 1;

                this->ranges(0, i) = r;
                this->ranges(1, i) = angle;
                this->ranges(2, i) = 0.0f;

                if (intensity != nullptr && msg.intensities.size() > i)
                    (*intensity)[i] = msg.intensities[i];
            }
        }

#if TKROS_VERSION == 1
        void fromRos(sensor_msgs::PointCloud2 &msg) {
            //convert
            sensor_msgs::PointCloud     tmp;
#endif
#if TKROS_VERSION == 2
        void fromRos(sensor_msgs::msg::PointCloud2 &msg) {
            //convert
            sensor_msgs::msg::PointCloud     tmp;
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::CLOUD; 

            // convert
            sensor_msgs::convertPointCloud2ToPointCloud(msg, tmp);

            // check if features is present
            tk::math::Vec<float>* intensity = nullptr;
            tk::math::Vec<float>* ring      = nullptr;
            tk::math::Vec<float>* angle     = nullptr;
            tk::math::Vec<float>* noise     = nullptr;
            tk::math::Vec<float>* time      = nullptr;
            if (this->features.exists(FEATURES_I)) 
                intensity   = &features[FEATURES_I]; 
            if (this->features.exists(FEATURES_CHANNEL)) 
                ring        = &features[FEATURES_CHANNEL]; 
            if (this->features.exists(FEATURES_ANGLE_VAR)) 
                angle       = &features[FEATURES_ANGLE_VAR]; 
            if (this->features.exists(FEATURES_NOISE)) 
                noise       = &features[FEATURES_NOISE];
            if (this->features.exists(FEATURES_TIME)) 
                time        = &features[FEATURES_TIME];


            int i_channel       = -1;
            int ring_channel    = -1;
            int noise_channel   = -1;
            int time_channel    = -1;
            if (intensity != nullptr || ring != nullptr) {
                for (int i = 0; i < tmp.channels.size(); i++) {
                    if (intensity != nullptr && tmp.channels[i].name == "intensity") {
                        i_channel = i;
                    }
                    if (ring != nullptr && tmp.channels[i].name == "ring") {
                        ring_channel    = i;                    
                    }
                    if (noise != nullptr && tmp.channels[i].name == "noise") {
                        noise_channel   = i;               
                    }
                    if (time != nullptr && tmp.channels[i].name == "time") {
                        time_channel    = i;               
                    }
                }
            }
            // fill points
            resize(tmp.points.size());
            for(int i = 0 ; i < tmp.points.size(); i++) {
                this->points(0, i) = tmp.points[i].x;
                this->points(1, i) = tmp.points[i].y;
                this->points(2, i) = tmp.points[i].z;
                this->points(3, i) = 1.0f;

                this->ranges(0, i) = std::sqrt(std::pow(tmp.points[i].x, 2) + std::pow(tmp.points[i].y, 2) + std::pow(tmp.points[i].z, 2));
                this->ranges(1, i) = std::acos(tmp.points[i].x / std::sqrt(std::pow(tmp.points[i].x, 2) + std::pow(tmp.points[i].y, 2))) * (tmp.points[i].y < 0 ? -1 : 1);
                this->ranges(2, i) = std::acos(tmp.points[i].z / this->ranges(0, i));

                if (intensity != nullptr && i_channel != -1)
                    (*intensity)[i]     = tmp.channels[i_channel].values[i];
                if (ring != nullptr && ring_channel != -1)
                    (*ring)[i]          = tmp.channels[ring_channel].values[i];
                if (angle != nullptr)
                    (*angle)[i]         = fmod((std::atan2(tmp.points[i].x, tmp.points[i].y) + M_PI_2) + M_PI*2, M_PI*2);
                if (noise != nullptr && noise_channel != -1)
                    (*noise)[i]         = tmp.channels[noise_channel].values[i];
                if (time != nullptr && time_channel != -1)
                    (*time)[i]          = tmp.channels[time_channel].values[i];
            }
        }
#endif

#ifdef PCL_ENABLED
        void toPcl(pcl::PointCloud<pcl::PointXYZI>::Ptr &aPclCloud) {
            aPclCloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
            aPclCloud->resize(this->size());
            
            bool hasI = false;
            tk::math::Vec<float>* intensity = nullptr;
            if (features.exists(tk::data::CloudData_gen::FEATURES_I)) {
                intensity   = &this->features[tk::data::CloudData::FEATURES_I];
                hasI        = true;
            }

            for(int i=0; i<aPclCloud->size(); ++i) {
                aPclCloud->points[i].x = this->points(0, i);
                aPclCloud->points[i].y = this->points(1, i);
                aPclCloud->points[i].z = this->points(2, i);
                if (hasI)
                    aPclCloud->points[i].intensity = (*intensity)[i];
            }
        }

        void fromPcl(const pcl::PointCloud<pcl::PointXYZI>::Ptr aPclCloud) {
            this->init();
            this->resize(aPclCloud->size());
            if (!features.exists(tk::data::CloudData_gen::FEATURES_I))
                this->addFeature(tk::data::CloudData::FEATURES_I);
            auto *intensity = &this->features[tk::data::CloudData::FEATURES_I];
            
            for(int i=0; i<aPclCloud->size(); ++i) {
                this->points(0, i) = aPclCloud->points[i].x;
                this->points(1, i) = aPclCloud->points[i].y;
                this->points(2, i) = aPclCloud->points[i].z;
                this->points(3, i) = 1;
                (*intensity)[i]         = aPclCloud->points[i].intensity;
            }
        }
#endif
    };
}}
