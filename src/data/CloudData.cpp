#include "tkCommon/data/CloudData.h"

namespace tk { namespace data {
    CloudData::CloudData() {
        this->nPoints = 0;
    }

    CloudData::~CloudData() {
        release();
    }

    void
    CloudData::init() {
        init(CLOUD_MAX_POINTS);
    }

    void
    CloudData::init(int size) {
        if (!isInitilized) {
            tkASSERT(size <= CLOUD_MAX_POINTS);

            this->points.resize(4, size);
            this->ranges.resize(2, size);
            this->features.resize(CLOUD_MAX_FEATURES, size);

            this->isInitilized = true;
        }
    }

    void
    CloudData::release() {
        if (this->isInitilized) {
            this->nPoints = 0;

            this->points.resize(0, 0);
            this->ranges.resize(0, 0);
            this->features.resize(0, 0);

            this->isInitilized = false;
        }
    }

    CloudData&
    CloudData::operator=(const CloudData& s) {
        SensorData::operator=(s);

        if (!this->isInitilized || this->points.rows() < s.points.rows())
            init();

        this->nPoints       = s.nPoints;
        std::memcpy(points.data(),   s.points.data(),   this->nPoints * 4 * sizeof(float));
        std::memcpy(ranges.data(),   s.ranges.data(),   this->nPoints * 2 * sizeof(float));
        std::memcpy(features.data(), s.features.data(), this->nPoints * CLOUD_MAX_FEATURES * sizeof(float));

        return *this;
    }
}}