#pragma once
#include "tkCommon/data/gen/CloudData_gen.h"

namespace tk { namespace data {

    class CloudData : public CloudData_gen{

        private:
            double lo_state_i = -1.0;
            double hi_state_i = -1.0;
        
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
            const auto fts = features.vals();
            for(int i=0; i<fts.size(); i++) {
                fts[i]->resize(n);
            }
        }

        int size() {
            return points.cols();
        }

        bool checkConsistency() {
            int n = size();
            if(n>0 && points.rows() != 4) return false;
            if(n != points.cols()) return false;
            for(int i=0; i<features.vals().size(); i++) {
                if(features.vals()[i]->size() != n)
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
                clsErr("Empty\n");
                    return;
            }

            if(!features.exists(tk::data::CloudData::FEATURES_I)){
                    clsErr("Error\n");
                    return;
            }
            const tk::math::Vec<float> *f = &features[tk::data::CloudData::FEATURES_I];

            // Get low and high value
            double lo = 99999.0f; 
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
            hi_state_i = 0.8f * hi_state_i + 0.2f * hi;

            // Apply gamma correction
            for(int i = 0; i < f->size(); i++){

                double value;
                value = (f->data_h[i] - lo) / (hi - lo);
                value = sqrt(value);
                if(value > 1.0f) value = 1.0f;
                if(value < 0.0f) value = 0.0f;
                f->data_h[i] = value;
            }
        }
    };
}}