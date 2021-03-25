#pragma once
#include "tkCommon/data/gen/CloudData_gen.h"

#ifdef ROS_ENABLED
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/LaserScan.h>
#include <sensor_msgs/point_cloud_conversion.h>
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

        void gammaCorrectionIntensity(){

            if(features.size() == 0){
                tkERR("Empty\n");
                    return;
            }

            if(!features.exists(tk::data::CloudData::FEATURES_I)){
                    tkERR("Error\n");
                    return;
            }
            tk::math::Vec<float> *intensity = &features[tk::data::CloudData::FEATURES_I];

            // Get low and high value
            /*double lo = 99999.0f; 
            double hi = -9999.0f;
            for(int i = 0; i < f->size(); i++){
                if(f->data_h[i] < lo)
                    lo = f->data_h[i];
                if(f->data_h[i] > hi)
                    hi = f->data_h[i];
            }

            // Calculate values for gamma correction
            lo *= 10.0f;
            hi /= 10.0f;
            if (lo_state_i < 0.0f) {
                lo_state_i = lo;
                hi_state_i = hi;
            }
            lo_state_i = 0.8f * lo_state_i + 0.2f * lo;
            hi_state_i = 0.8f * hi_state_i + 0.2f * hi;*/



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
                tkERR("Error creating file\n");
            }   

            bool hasF = false;
            if (features.exists(tk::data::CloudData_gen::FEATURES_I)) {
                hasF = true;
            }  

            tk::math::Vec4f point;
            for(int i = 0; i < points.cols(); i++){
                point[0] = points(0,i);
                point[1] = points(1,i);
                point[2] = points(2,i);
                if(hasF){
                    auto intensity = &features[tk::data::CloudData_gen::FEATURES_I];
                    point[3] = (*intensity)[i];
                }else{
                    point[3] = 0;
                }
                output.write((char *) &point.cpu.data, 4*sizeof(float));
            }
            output.close();
        }

        void fromKitty(const std::string& fileName){
            std::fstream input(fileName.c_str(), std::ios::in | std::ios::binary);
            if(!input.good()){
                tkERR("Error loading file\n");
            }
            input.seekg(0, std::ios::beg);           

            std::vector<tk::math::Vec4f> data;            
            for (int i = 0; input.good() && !input.eof(); i++) {
                tk::math::Vec4f point;
                input.read((char *) &point.cpu.data, 4*sizeof(float));
                data.push_back(point);
            }
            input.close();

            if (!features.exists(tk::data::CloudData_gen::FEATURES_I)) {
                features.add(tk::data::CloudData_gen::FEATURES_I);
            } 
            resize(data.size());

            auto intensity = &features[tk::data::CloudData_gen::FEATURES_I];
            for(int i = 0; i < data.size(); i++){
                points(0,i) = data[i][0];
                points(1,i) = data[i][1];
                points(2,i) = data[i][2];
                points(3,i) = 1;
                (*intensity)[i] = data[i][3];
            }
        }

#ifdef ROS_ENABLED
        void toRos(sensor_msgs::LaserScan &msg) {
            this->header.toRos(msg.header);
            
            /*
            float max_range, min_range, max_angle, min_angle, angle_increment;

            msg.ranges.resize(size());
            msg.intensities.resize(size());
            for (int i = 0; i < size(); i++) {
                msg.ranges[i] = this->ranges(0, 1);
            }
            */
            tkWRN("Not implemented.\n");
        }

        void toRos(sensor_msgs::PointCloud2 &msg) {
            this->header.toRos(msg.header);
            /*
            sensor_msgs::PointCloud tmp;
            tk::math::Vec<float>*   intensity = nullptr;
            if (this->features.exists(FEATURES_I))
                tk::math::Vec<float>* intensity = &features[FEATURES_I];

            // fill points
            tmp.points.resize(size());
            if (intensity != nullptr) {
                tmp.channels.resize(1);
                tmp.channels[0].name = "intensity";
            }
            for(int i = 0 ; i < size(); i++) {
                tmp.points[i].x = this->points(0, i); 
                tmp.points[i].y = this->points(1, i); 
                tmp.points[i].z = this->points(2, i); 

                if (intensity != nullptr)
                    tmp.channels[0].values[i] = (*intensity)[i];
            }

            // convert
            sensor_msgs::convertPointCloudToPointCloud2(tmp, msg);
            */

            msg.width  = size();
            msg.height = 1;
            msg.is_bigendian = false;
            msg.is_dense = true;

            //fields setup
            sensor_msgs::PointCloud2Modifier modifier(msg);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            msg.point_step = addPointField(msg, "intensity", 1, sensor_msgs::PointField::FLOAT32, msg.point_step);
            msg.point_step = addPointField(msg, "ring",      1, sensor_msgs::PointField::UINT16,  msg.point_step);
            msg.point_step = addPointField(msg, "time",      1, sensor_msgs::PointField::FLOAT32, msg.point_step);
            msg.row_step = msg.width * msg.point_step;
            msg.data.resize(msg.height * msg.row_step);
            
            sensor_msgs::PointCloud2Iterator<float>    iter_x(msg, "x");
            sensor_msgs::PointCloud2Iterator<float>    iter_y(msg, "y");
            sensor_msgs::PointCloud2Iterator<float>    iter_z(msg, "z");
            sensor_msgs::PointCloud2Iterator<float>    iter_intensity(msg, "intensity");
            sensor_msgs::PointCloud2Iterator<uint16_t> iter_ring(msg, "ring");
            sensor_msgs::PointCloud2Iterator<float>    iter_time(msg, "time");
            tk::math::Vec<float> *fI = &features[tk::data::CloudData_gen::FEATURES_I];
            tk::math::Vec<float> *fC = &features[tk::data::CloudData_gen::FEATURES_CHANNEL];
            tk::math::Vec<float> *fT = &features[tk::data::CloudData_gen::FEATURES_TIME];

            for(int i=0; i<size(); i++) {
                *iter_x         = points(0,i);     ++iter_x; 
                *iter_y         = points(1,i);     ++iter_y; 
                *iter_z         = points(2,i);     ++iter_z; 
                *iter_intensity = (*fI)[i]; ++iter_intensity; 
                *iter_ring      = (*fC)[i]; ++iter_ring;  
                *iter_time      = (*fT)[i]; ++iter_time;  
            }

        }

        void fromRos(sensor_msgs::LaserScan &msg) {
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

        void fromRos(sensor_msgs::PointCloud2 &msg) {
            this->header.fromRos(msg.header);
            this->header.type   = DataType::CLOUD; 

            // convert
            sensor_msgs::PointCloud     tmp;
            sensor_msgs::convertPointCloud2ToPointCloud(msg, tmp);

            // check if intensity is present
            tk::math::Vec<float>* intensity = nullptr;
            int i_channel = -1;
            if (this->features.exists(FEATURES_I)) {
                tk::math::Vec<float>* intensity = &features[FEATURES_I]; 
                for (int i = 0; i < tmp.channels.size(); i++) {
                    if (tmp.channels[i].name == "intensity") {
                        i_channel = i;
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
                    (*intensity)[i] = tmp.channels[i_channel].values[i];
            }
        }
#endif
    };
}}