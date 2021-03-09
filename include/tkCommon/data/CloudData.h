#pragma once
#include "tkCommon/data/gen/CloudData_gen.h"

#ifdef ROS_ENABLED
    #include <sensor_msgs/PointCloud2.h>
    #include <sensor_msgs/point_cloud2_iterator.h>
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

        #ifdef ROS_ENABLED
        bool toMsg(sensor_msgs::PointCloud2 &cloud) {
            ros::Time ts;
            std_msgs::Header header;
            header.frame_id = "lidar";
            header.stamp = ts.fromNSec(this->header.stamp*1e3);

            cloud.header = header;
            cloud.width  = size();
            cloud.height = 1;
            cloud.is_bigendian = false;
            cloud.is_dense = true;

            //fields setup
            sensor_msgs::PointCloud2Modifier modifier(cloud);
            modifier.setPointCloud2FieldsByString(1, "xyz");
            cloud.point_step = addPointField(cloud, "intensity", 1, sensor_msgs::PointField::FLOAT32, cloud.point_step);
            cloud.point_step = addPointField(cloud, "ring",      1, sensor_msgs::PointField::UINT16,  cloud.point_step);
            cloud.point_step = addPointField(cloud, "time",      1, sensor_msgs::PointField::FLOAT32, cloud.point_step);
            cloud.row_step = cloud.width * cloud.point_step;
            cloud.data.resize(cloud.height * cloud.row_step);
            
            sensor_msgs::PointCloud2Iterator<float>    iter_x(cloud, "x");
            sensor_msgs::PointCloud2Iterator<float>    iter_y(cloud, "y");
            sensor_msgs::PointCloud2Iterator<float>    iter_z(cloud, "z");
            sensor_msgs::PointCloud2Iterator<float>    iter_intensity(cloud, "intensity");
            sensor_msgs::PointCloud2Iterator<uint16_t> iter_ring(cloud, "ring");
            sensor_msgs::PointCloud2Iterator<float>    iter_time(cloud, "time");
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
        #endif

    };
}}