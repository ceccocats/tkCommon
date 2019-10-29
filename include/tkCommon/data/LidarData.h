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

    //Velodyne Data
    /////////////////////////////////////////////////////////////////////////////////////////////////////////

    /* Values for VLP16 VLP32c*/
    //-----------------------------------
    const int LASER_PER_FIRING  = 32;
    const int FIRING_PER_PKT    = 12;
    const int NUM_ROT_ANGLES    = 36001;
    const int MAX_NUM_LASERS    = 64;

    // Default port of lidar
    const int DEFAULT_PORT      = 2368;

    // Numer of max point per spin
    const int N_POINT_MAX       = 59904;

    const int PACKET_FOR_SPIN_VLP_32   = 156;
    const int PACKET_FOR_SPIN_VLP_16   = 78;
    //------------------------------------

    const int RX_BUFFER         = 1500;

    struct LaserCorrection
    {
        double azimuthCorrection;
        double verticalCorrection;
        double sinVertCorrection;
        double cosVertCorrection;
        double sinVertOffsetCorrection;
    };

    #pragma pack(push, 1)
    typedef struct LaserReturn
    {
        unsigned short distance;
        unsigned char intensity;
    } LaserReturn;
    #pragma pack(pop)

    struct FiringData
    {
        unsigned short blockIdentifier;
        unsigned short rotationalPosition;
        LaserReturn laserReturns[LASER_PER_FIRING];
    };

    struct DataPacket
    {
        FiringData firingData[FIRING_PER_PKT];
        unsigned int  gpsTimestamp;
        unsigned char blank1;
        unsigned char blank2;
    };

    enum Block
    {
        BLOCK_0_TO_31 = 0xeeff,
        BLOCK_32_TO_63 = 0xddff
    };
    
}}