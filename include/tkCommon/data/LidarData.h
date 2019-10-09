#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    static const int LIDAR_MAX_POINTS 200000

    struct LidarData_t {

        /**
         * @brief Points in lidar Pointcloud
         * 
         */
        int                                                             nPoints;
        
        /**
         * @brief Pointcloud
         *
         *  ( X )
         *  ( Y )
         *  ( Z )
         *  ( 1 )
         * 
         */
        Eigen::Matrix<float  , 4, LIDAR_MAX_POINTS, Eigen::ColMajor>    points;

        /**
         * @brief Points intensity
         * 
         */
        Eigen::Matrix<float  , 1, LIDAR_MAX_POINTS, Eigen::ColMajor>    intensity;

        /**
         * @brief Overloading for struct copy
         * 
         */
        LidarData_t& operator=(const LidarData_t& s){

            nPoints = s.nPoints;
            std::memcpy(points.data(),      s.points.data(),    nPoints * 4 * sizeof(float) );
            std::memcpy(intensity.data(),   s.intensity.data(), nPoints * sizeof(float) );
        }
    };

}}