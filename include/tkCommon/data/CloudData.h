#pragma once
#include "tkCommon/data/DataHeader.h"

namespace tk { namespace data {
    static const int CLOUD_MAX_POINTS       = 200000;
    static const int CLOUD_MAX_FEATURES     = 16;
    enum CloudFeatureType {
        R              = 0,
        G              = 1,
        B              = 2,
        I              = 3,
        CLASS          = 4,
        VELOCITY       = 5,
        FALSE_DET      = 6,
        RANGE_VAR      = 7,
        VELOCITY_VAR   = 8,
        ANGLE_VAR      = 9,
        RCS            = 10,
        ID             = 11
    };

    struct CloudData_t {
        DataHeader_t        header;

        int                 nPoints;
        Eigen::MatrixXf     points;
        Eigen::MatrixXf     ranges;
        Eigen::MatrixXf     features;


        void init(){
            this->points.resize(4, CLOUD_MAX_POINTS);
            this->ranges.resize(2, CLOUD_MAX_POINTS);
            this->features.resize(CLOUD_MAX_FEATURES, CLOUD_MAX_POINTS);
        }

        /**
         * @brief Overloading for struct copy
         *
         */
        CloudData_t& operator=(const CloudData_t& s){
            init();

            nPoints     = s.nPoints;
            header      = s.header;
            std::memcpy(points.data(),   s.points.data(),   nPoints * 4 * sizeof(float));
            std::memcpy(ranges.data(),   s.ranges.data(),   nPoints * 2 * sizeof(float));
            std::memcpy(features.data(), s.features.data(), nPoints * CLOUD_MAX_FEATURES * sizeof(float));

            return *this;
        }
    };
}}