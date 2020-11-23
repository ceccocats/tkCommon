#pragma once
#include "tkCommon/data/gen/CalibData_gen.h"
#include "tkCommon/utils.h"

namespace tk{ namespace data{

    class CalibData : public CalibData_gen{
        public:

        void init(){
            CalibData_gen::init();

            k.resize(3,3);
            for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
            k(i,j) = 0;
            k(0,0) = 1;
            k(1,1) = 1;
            k(2,2) = 1;
            d.resize(1,5);
            for(int i = 0; i < 5; i++)
            d(0,i) = 0;
            r.resize(3,3);
            for(int i = 0; i < 3; i++)
            for(int j = 0; j < 3; j++)
            r(i,j) = 0;
            r(0,0) = 1;
            r(1,1) = 1;
            r(2,2) = 1;

        }

        /**
         * @brief       Method that load a calibartion file
         * 
         * @param conf  yaml node config
         * 
         *              K           |D          |R      |
         *              :----------:|:---------:|:---------:|
         *              float(9)    |float(5)   |float(9)   |
         * @return status
         */
        bool load(const YAML::Node conf){

            init();

            //std::vector<float> k_tmp = getYAMLconf<std::vector<float>>(conf, "K", std::vector<float>(9,0));
            std::vector<float> k_tmp = conf["K"]? conf["K"].as<std::vector<float>>(): std::vector<float>(9,0);
            tkASSERT(k_tmp.size() == 9)
            for(int i = 0; i < 9; i++)
                k[i] = k_tmp[i];

            std::vector<float> d_tmp = conf["D"]? conf["D"].as<std::vector<float>>(): std::vector<float>(5,0);
            tkASSERT(d_tmp.size() >= 5)
            for(int i = 0; i < 5; i++)
                d[i] = d_tmp[i];

            std::vector<float> r_tmp = conf["R"]? conf["R"].as<std::vector<float>>(): std::vector<float>(9,0);
            tkASSERT(r_tmp.size() == 9)
            for(int i = 0; i < 9; i++)
                r[i] = r_tmp[i];

            return true;
        }

        void world2cam(tk::common::Tfpose &tf, tk::math::Mat<float> &point, tk::math::Mat<float> &dst){

            Eigen::MatrixXf t = tf.matrix() * point.matrix();
            t.conservativeResize(3, point.cols());

            Eigen::MatrixXf result = k.matrix() * t;

            dst.resize(3, point.cols());

            dst.matrix() = result;

        }

        void resize(float scale){

            k(0,0) *= scale;
            k(1,1) *= scale;
            k(0,2) *= scale;
            k(1,2) *= scale;

        }

    };
}}