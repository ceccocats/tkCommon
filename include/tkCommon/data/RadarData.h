#pragma once
#include "tkCommon/data/CloudData.h"

namespace tk { namespace data {
    static const int CONTINENTAL_MAX_RADAR = 8;

    struct RadarData_t {
        DataHeader_t    header;
        int             nRadar;

        std::vector<tk::data::CloudData_t>  near_data;
        std::vector<tk::data::CloudData_t>  far_data;

        void init() {
            near_data.resize(CONTINENTAL_MAX_RADAR);
            far_data.resize(CONTINENTAL_MAX_RADAR);
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