#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    static const int LIDAR_MAX_POINTS = 200000;

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
        Eigen::Matrix<float  , 1, LIDAR_MAX_POINTS, Eigen::RowMajor>    intensity;

        /**
         * @brief ID matrix
         * 
         */
        Eigen::MatrixXi                                                 idMatrix;

        /**
         * @brief Overloading for struct copy
         * 
         */
        LidarData_t& operator=(const LidarData_t& s){

            nPoints = s.nPoints;
            std::memcpy(points.data(),      s.points.data(),    nPoints * 4 * sizeof(float) );
            std::memcpy(intensity.data(),   s.intensity.data(), nPoints * sizeof(float) );
            idMatrix = s.idMatrix;
            return *this;
        }
    };

    //Velodyne Data
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace velodyne{

        /* Values for VLP16 VLP32c*/
        //-----------------------------------
        const int LASER_PER_FIRING         = 32;
        const int FIRING_PER_PKT           = 12;
        const int NUM_ROT_ANGLES           = 36001;
        const int MAX_NUM_LASERS           = 64;

        // Default port of lidar
        const int DEFAULT_PORT             = 2368;

        // Numer of max point per spin
        const int N_POINT_MAX              = 59904;

        const int PACKET_FOR_SPIN_VLP_32   = 156;
        const int PACKET_FOR_SPIN_VLP_16   = 78;
        //------------------------------------

        const int RX_BUFFER                = 1500;

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
    
    }

    //Ouster Data
    /////////////////////////////////////////////////////////////////////////////////////////////////////////
    namespace ouster{

        #pragma pack(push, 1)
        typedef struct dataPacket{
            uint32_t    range;
            uint16_t    reflectivity;
            uint16_t    signal_photons;
            uint16_t    noise_photons;
            uint16_t    padding;
        }dataPacket;
        #pragma pack(pop)

        #pragma pack(push, 1)
        typedef struct ousterPacket{
            uint64_t    timestamp;
            uint16_t    column_id;
            uint16_t    frame_id;
            uint32_t    encoder_count;
            dataPacket  data[64];   //TODO: 128 is 128?
            uint32_t    valid;

        }ousterPacket;
        #pragma pack(pop)

        #pragma pack(push, 1)
        typedef struct imuData{
            uint64_t    imu_sys_timestamp;
            uint64_t    imu_acc_timestamp;
            uint64_t    imu_gyr_timestamp;

            float       imu_la_x;
            float       imu_la_y;
            float       imu_la_z;

            float       imu_av_x;
            float       imu_av_y;
            float       imu_av_z;

        }imuData;
        #pragma pack(pop)

        const int    LIDAR_PORT          = 7502;
        const int    LIDAR_LEN_PACKET    = 12608; //(bytes)
        const int    N_POINT_MAX         = 131072;

        const int    IMU_PORT            = 7503;
        const int    IMU_LEN_PACKET      = 48; //(bytes)




        const float lidar_to_sensor_transform[16]   ={  -1,                 0,                  0,                  0,                  0,  
                                                        -1,                 0,                  0,                  0,                  0, 
                                                        1,                  36.18,              0,                  0,                  0, 
                                                        1
                                                    };

        const float imu_to_sensor_transform[16]     ={  1,                  0,                  0,                  6.253,              0, 
                                                        1,                  0,                  -11.775,            0,                  0, 
                                                        1,                  7.645,              0,                  0,                  0, 
                                                        1
                                                    };

        const float beam_altitude_angles[64]        ={  17.283,             16.686,             16.117,             15.56,              15.096,
                                                        14.524,             3.956,              13.425,             12.96,              12.399, 
                                                        11.836,             11.306,             10.855,             10.291,             9.749000000000001, 
                                                        9.217000000000001,  8.759,              8.207000000000001,  7.669,              7.119, 
                                                        6.674,              6.131,              5.58,               5.045,              4.595, 
                                                        4.047,              3.505,              2.964,              2.517,              1.973, 
                                                        1.439,              0.88,               0.442,              -0.102,             -0.648, 
                                                        -1.194,             -1.637,             -2.181,             -2.721,             -3.293, 
                                                        -3.719,             -4.271,             -4.813,             5.369,              -5.812, 
                                                        -6.345,             -6.9,               -7.456,             -7.9,               -8.436, 
                                                        -8.991,             -9.558999999999999, -10.004,            -10.546,            -11.094, 
                                                        -11.675,            -12.122,            12.655,             -13.227,            -13.813, 
                                                        -14.273,            -14.82,             -15.391,            -15.994
                                                    }; 
                                                        
        const float beam_azimuth_angles[64]         ={  3.04,               0.877,              -1.272,             -3.413,             3.026, 
                                                        0.887,              -1.242,             -3.356,             3.024,              0.889, 
                                                        -1.202,             -3.297,             3.01,               0.927,              1.182, 
                                                        -3.257,             3.026,              0.9360000000000001, -1.141,             -3.214, 
                                                        3.037,              0.963,              -1.114,             -3.185,             3.051, 
                                                        0.99,               -1.082,             -3.147,             3.088,              1.011, 
                                                        -1.068,             -3.127,             3.107,              1.032,              -1.04, 
                                                        -3.099,             3.134,              1.052,              -1.015,             -3.073, 
                                                        3.16,               1.08,               -0.978,             -3.056,             3.194, 
                                                        1.116,              -0.951,             -3.05,              3.231,              1.144, 
                                                        -0.9350000000000001,-3.028,             3.282,              1.182,              -0.92, 
                                                        -3.027,             3.314,              1.209,              -0.888,             -3.025, 
                                                        3.39,               1.256,              -0.877,             -3.018
                                                    };


    }


}}