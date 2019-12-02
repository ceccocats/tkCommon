#pragma once
#include "tkCommon/data/CloudData.h"

namespace tk { namespace data {

    class LidarData : public CloudData{
    
    public:

        static const int LIDAR_MAX_POINTS = 200000;

        /**
         * @brief Points intensity
         */
        Eigen::MatrixXf intensity;

        /**
         * @brief ID matrix
         */
        Eigen::MatrixXi idMatrix;

        /**
         * @brief Initialization method only for Eigen points and intensity.
         */
        void init(){
            SensorData::init();
            this->points.resize(4,LIDAR_MAX_POINTS);
            this->intensity.resize(1,LIDAR_MAX_POINTS);
        }
        /**
         * @brief Initialization method for all data
         * 
         * @param o_layers horizzontal layers
         * @param v_layers vertical layers
         */
        void init(int o_layers, int v_layers){
            this->init();
            this->initIdMatrix(v_layers,o_layers);
        }
        /**
         * @brief Initialization method for idMatrix and setting to -1
         * 
         * @param o_layers horizzontal layers
         * @param v_layers vertical layers
         */
        void initIdMatrix(int o_layers, int v_layers){
            this->idMatrix.resize(v_layers,o_layers);
            this->idMatrix.setConstant(v_layers,o_layers,-1);
        }
        /**
         * @brief release method
         * 
         */
        void release(){
            this->points.resize(0,0);
            this->intensity.resize(0,0);
            this->idMatrix.resize(0,0);            
        }
        /**
         * @brief Overloading of operator = for class copy.
         * 
         */
        LidarData& operator=(const LidarData& s){
            this->nPoints   = s.nPoints;
            this->idMatrix  = s.idMatrix;
            this->points    = s.points;
            this->intensity = s.intensity;
            return *this;
        }

        bool checkDimension(SensorData *s){
            return true;
        }
    };
}}