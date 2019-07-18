#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    #define LIDAR_MAX_POINTS 119808

    struct LidarData_t {
        int                                                             nPoints;
        Eigen::Matrix<float  , 4, LIDAR_MAX_POINTS, Eigen::ColMajor>    points;
        Eigen::Matrix<uint8_t, 1, LIDAR_MAX_POINTS, Eigen::ColMajor>    intensity;

        LidarData_t& operator=(const LidarData_t& s){

            nPoints = s.nPoints;
            std::memcpy(points.data(),      s.points.data(),    nPoints * 4 * sizeof(float));
            std::memcpy(intensity.data(),   s.intensity.data(), nPoints * sizeof(uint8_t));
        }
    };

}}