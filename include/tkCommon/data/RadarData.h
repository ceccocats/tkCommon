#pragma once
#include "tkCommon/common.h"

namespace tk { namespace data {

    #define RADAR_MAX_POINTS 5000
    #define N_RADAR 6

    #pragma pack(push, 1)
    struct RadarData_t {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        tk::common::TimeStamp   near_stamp[N_RADAR];
        tk::common::TimeStamp   far_stamp[N_RADAR];
        int                     near_n_points[N_RADAR];
        int                     far_n_points[N_RADAR];
        Eigen::Matrix<float, 4, RADAR_MAX_POINTS, Eigen::ColMajor> near_points[N_RADAR];
        Eigen::Matrix<float, 4, RADAR_MAX_POINTS, Eigen::ColMajor> far_points[N_RADAR];
        //matrice feature
    };
    #pragma pack(pop)

}}
