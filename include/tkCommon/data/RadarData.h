#pragma once

#include "tkCommon/data/CloudData.h"

namespace tk { namespace data {
    static const int MAX_RADAR = 8;

    struct RadarData_t {
        tk::data::DataHeader_t      header;
        int                         nRadar;

        std::vector<tk::data::CloudData_t>  near_data;
        std::vector<tk::data::CloudData_t>  far_data;


        void init() {
            near_data.resize(MAX_RADAR);
            far_data.resize(MAX_RADAR);

            for (int i = 0; i < MAX_RADAR; i++) {
                near_data[i].init();
                far_data[i].init();
            }
        }

        /**
         * @brief Overloading for struct copy
         *
         */
        RadarData_t& operator=(const RadarData_t& s){
            init();

            this->nRadar    = s.nRadar;
            this->header    = s.header;
            for (int i = 0; i < nRadar; i++) {
                this->near_data[i]    = s.near_data[i];
                this->far_data[i]     = s.far_data[i];
            }

            return *this;
        }
    };
}}