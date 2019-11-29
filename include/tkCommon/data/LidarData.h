#pragma once
#include <tkCommon/data/DataHeader.h>

namespace tk { namespace data {

    static const int LIDAR_MAX_POINTS = 200000;

    struct LidarData_t {

        tk::data::DataHeader_t header;

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
        Eigen::MatrixXf                                                  points;

        /**
         * @brief Points intensity
         * 
         */
        Eigen::MatrixXf                                                 intensity;

        /**
         * @brief ID matrix
         * 
         */
        Eigen::MatrixXi                                                 idMatrix;

        void init(){

            this->points.resize(4,LIDAR_MAX_POINTS);
            this->intensity.resize(1,LIDAR_MAX_POINTS);
        }

        /**
         * @brief Overloading for struct copy
         * 
         */
        LidarData_t& operator=(const LidarData_t& s){
            init(); // it allocates only if it is necessary

            nPoints = s.nPoints;
            std::memcpy(points.data(),      s.points.data(),    nPoints * 4 * sizeof(float) );
            std::memcpy(intensity.data(),   s.intensity.data(), nPoints * sizeof(float) );
            idMatrix = s.idMatrix;
            return *this;
        }
    };

    struct replayPcap_t{

        float           velocity      = 1;
        bool            pressedStart  = false;
        bool            pressedStop   = false;
        bool            pressedBar    = false;
        int             barNumPacket  = 0;
        int             barMinVal     = 0;
        int             barMaxVal     = 0;
        std::string     textOutput    = ""; 
    };

}}