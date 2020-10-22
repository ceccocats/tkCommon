#pragma once
#include "tkCommon/data/gen/CloudData_gen.h"

namespace tk { namespace data {

    class CloudData : public CloudData_gen{

        private:
            double lo_state_i = -1.0;
            double hi_state_i = -1.0;
        
        public:

        void printFeaturesMap() {
            std::cout<<"Cloud Avaible features: \n";
            for(auto i : features) {
                std::cout<<"\t"<<i.first<<": "<<i.second<<"\n";
            }
        }

        void gammaCorrectionIntensity(){

            if(features.size() == 0){
                clsErr("Empty\n");
                    return;
            }

            if(features.count(tk::data::CloudData::FEATURES_I) == 0){
                    clsErr("Error\n");
                    return;
            }
            tk::math::Vec<float> *f = &features[tk::data::CloudData::FEATURES_I];

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