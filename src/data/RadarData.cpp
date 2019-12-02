#include "tkCommon/data/RadarData.h"

namespace tk { namespace data {
    RadarData::RadarData() {
        nRadar = 0;
    }

    void
    RadarData::init() {
        if (!this->isInitilized) {
            this->near_data.resize(RADAR_MAX);
            this->far_data.resize(RADAR_MAX);

            for (int i = 0; i < RADAR_MAX; i++) {
                if (!this->near_data[i].isInitilized)
                    this->near_data[i].init(RADAR_MAX_PONINTS);
                if (!this->far_data[i].isInitilized)
                    this->far_data[i].init(RADAR_MAX_PONINTS);
            }

            this->isInitilized = true;
        }
    }

    void
    RadarData::release() {
        if (this->isInitilized) {
            for (int i = 0; i < RADAR_MAX; i++) {
                this->near_data[i].release();
                this->far_data[i].release();
            }

            this->near_data.clear();
            this->far_data.clear();
            this->nRadar        = 0;
            this->isInitilized  = false;
        }
    }

    RadarData&
    RadarData::operator=(const RadarData& s){
        SensorData::operator=(s);

        if (!this->isInitilized)
            init();

        this->nRadar    = s.nRadar;
        for (int i = 0; i < nRadar; i++) {
            this->near_data[i]    = s.near_data[i];
            this->far_data[i]     = s.far_data[i];
        }

        return *this;
    }
}}