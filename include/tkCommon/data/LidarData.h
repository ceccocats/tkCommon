#pragma once
#include "tkCommon/data/CloudData.h"
#include "tkCommon/gui/shader/pointcloud4f.h"

namespace tk { namespace data {

    class LidarData : public CloudData{
    
    public:

        /**
         * @brief Points intensity
         */
        Eigen::MatrixXf intensity;

        /**
         *  Laser ID
         */
        Eigen::MatrixXf laserID;


        /**
         * @brief ID matrix
         */
        Eigen::MatrixXi idMatrix;


        /**
         * @brief PolarCloud
         */
        Eigen::MatrixXi polarCloud;

        /**
         * @brief Initialization method only for Eigen points and intensity.
         */
        void init(){
            SensorData::init();
            this->nPoints = 0;
            this->points.resize(4,CLOUD_MAX_POINTS);
            this->intensity.resize(1,CLOUD_MAX_POINTS);
            this->laserID.resize(1,CLOUD_MAX_POINTS);
            this->idMatrix.resize(0,0);

            this->points.setZero();
            this->intensity.setZero();
            this->laserID.setZero();
            this->idMatrix.setZero();

            header.sensor = sensorName::LIDAR;
        }
        /**
         * @brief Initialization method for all data
         * 
         * @param o_layers horizzontal layers
         * @param v_layers vertical layers
         */
        void init(int o_layers, int v_layers){
            this->init();
            this->initIdMatrix(o_layers,v_layers);
        }
        /**
         * @brief Initialization method for idMatrix and setting to -1
         * 
         * @param o_layers horizzontal layers
         * @param v_layers vertical layers
         */
        void initIdMatrix(int o_layers, int v_layers){
            this->idMatrix.resize(o_layers,v_layers);
            this->idMatrix.setConstant(o_layers,v_layers,-1);
            header.name = sensorName::LIDAR;
        }
        /**
         * @brief Initialization method for idMatrix and setting to -1
         * 
         * @param o_layers horizzontal layers
         * @param v_layers vertical layers
         */
        void initPolarCloud(int n, int dim){
            this->idMatrix.resize(n,dim);
            this->idMatrix.setConstant(n,dim,-1);
            header.name = sensorName::LIDAR;
        }
        /**
         * @brief release method
         * 
         */
        void release(){
            this->points.resize(0,0);
            this->intensity.resize(0,0);
            this->laserID.resize(0,0);
            this->idMatrix.resize(0,0);            
        }
        /**
         * @brief Overloading of operator = for class copy.
         * 
         */
        LidarData& operator=(const LidarData& s){
            SensorData::operator=(s);

            this->nPoints   = s.nPoints;
            this->idMatrix  = s.idMatrix;
            this->points    = s.points;
            this->intensity = s.intensity;
            this->laserID   = s.laserID;
            return *this;
        }

        bool checkDimension(SensorData *s){
            return true;
        }

    private:

        bool initted = false;
        tk::gui::shader::pointcloud4f   shader;
        tk::gui::Buffer<float>          gpu_cloud;


    public:

        void onAdd(tk::gui::Viewer *viewer){

            if(initted == false){
                initted = true;
                shader.init();
                gpu_cloud.init();
            }

            gpu_cloud.setData(points.data(),nPoints*4);
        }

        void draw(tk::gui::Viewer *viewer){

            shader.draw(&gpu_cloud,nPoints);			
        }

        ~LidarData(){
            shader.close();
            gpu_cloud.release();
        }
    };
}}