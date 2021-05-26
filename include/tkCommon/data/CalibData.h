#pragma once
#include "tkCommon/data/gen/CalibData_gen.h"
#include "tkCommon/utils.h"
#include "tkCommon/common.h"

namespace tk{ namespace data{

    class CalibData : public CalibData_gen{
        public:

        tk::common::Tfpose tf;

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

        float fx() {return k(0,0);}
        float fy() {return k(1,1);}
        float cx() {return k(2,0);}
        float cy() {return k(2,1);}

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

            w = conf["width"]? conf["width"].as<int>() : w;
            h = conf["height"]? conf["height"].as<int>() : h;

            //std::vector<float> k_tmp = getYAMLconf<std::vector<float>>(conf, "K", std::vector<float>(9,0));
            std::vector<float> k_tmp = conf["K"]? conf["K"].as<std::vector<float>>(): std::vector<float>(9,0);
            tkASSERT(k_tmp.size() == 9 || k_tmp.size() == 4);
            if(k_tmp.size() == 9)
                for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
                        k[i*3+j] = k_tmp[i*3 + j];
                        //k(i,j) = k_tmp[i*3 + j];
            else if(k_tmp.size() == 4){
                //for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
                //        k[i*3+j] = i==j?1:0;
                k[0] = k_tmp[0]; // 0,0
                k[4] = k_tmp[1]; // 1,1
                k[2] = k_tmp[2]; // 0,2
                k[5] = k_tmp[3]; // 1,2
            }


            std::vector<float> d_tmp = conf["D"]? conf["D"].as<std::vector<float>>(): std::vector<float>(5,0);
            tkASSERT(d_tmp.size() >= 4)
            for(int i = 0; i < d_tmp.size(); i++)
                d[i] = d_tmp[i];

            std::vector<float> r_tmp = conf["R"]? conf["R"].as<std::vector<float>>(): std::vector<float>(9,0);
            tkASSERT(r_tmp.size() == 9)
            for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++)
                    r[i*3+j] = r_tmp[i*3 + j];
                    //r(i,j) = r_tmp[i*3 + j];
    
            std::vector<tk::common::Tfpose> tfs = tk::common::YAMLreadTf(conf["tf"]);
            if(!tfs.empty())
                this->tf = tfs[0];

            return true;
        }

        void save(std::string file){

            std::vector<float> K(4);
            K[0] = fx();
            K[1] = fy();
            K[2] = cx();
            K[3] = cy();

            std::vector<float> D(5);
            for(int i = 0; i < 5; i++) D[i] = d[i];
            
            std::vector<float> R(9);
            for(int i = 0; i < 9; i++) R[i] = r[i];

            tk::common::Vector3<float> pos = tk::common::tf2pose(tf);
            tk::common::Vector3<float> rot = tk::common::tf2pose(tf);

            std::vector<float> tf_vec(6);
            tf_vec[0] = pos.x();
            tf_vec[1] = pos.y();
            tf_vec[2] = pos.z();
            tf_vec[3] = rot.x() * 180.0f / M_PI;
            tf_vec[4] = rot.y() * 180.0f / M_PI;
            tf_vec[5] = rot.z() * 180.0f / M_PI;

            YAML::Emitter out;
            out << YAML::BeginMap;
            out << YAML::Key << "K";
            out << YAML::Value << YAML::Flow << K;
            out << YAML::Key << "D";
            out << YAML::Value << YAML::Flow << D;
            out << YAML::Key << "R";
            out << YAML::Value << YAML::Flow << R;
            out << YAML::Key << "tf";
            out << YAML::Value << YAML::Flow << tf_vec;
            out << YAML::Key << "width";
            out << YAML::Value << w;
            out << YAML::Key << "height";
            out << YAML::Value << h;
            out << YAML::EndMap;

            std::ofstream fout(file);

            fout<<out.c_str();

            fout.close();

        }

        void world2cam(tk::common::Tfpose &tf, tk::math::Mat<float> &point, tk::math::Mat<float> &dst){
            Eigen::Matrix3f fix;

            fix << 0, -1,  0,
                   0,  0, -1,
                   1,  0,  0;
            //tk::common::Tfpose lol; lol.linear() = fix;
            //std::cout<<tk::common::tf2rot(lol)<<"\n";

            Eigen::MatrixXf t = (tf.inverse().matrix()) * point.matrix();
            t.conservativeResize(3, point.cols());

            t = fix * t;

            Eigen::MatrixXf result = k.matrix().transpose() * t;
            dst.copyFrom(result.data(), result.rows(), result.cols());
        }

        void world2pix(tk::common::Tfpose &tf, tk::math::Mat<float> &point, tk::math::Mat<float> &dst){
            

            world2cam(tf, point, dst);
            for(int i=0; i<dst.cols(); i++) {
                if(dst(2,i) > 0){
                    dst(0,i) = dst(0,i) / dst(2,i);
                    dst(1,i) = dst(1,i) / dst(2,i);
                    dst(2,i) = 1.0;
                } else{
                    dst(0,i) = 0.0;
                    dst(1,i) = 0.0;
                    dst(2,i) = 0.0;
                }
                
            }
        }

        void rescale(float scale){
            w *= scale;
            h *= scale;
            k[0*3+0] *= scale;
            k[1*3+1] *= scale;
            k[0*3+2] *= scale;
            k[1*3+2] *= scale;

        }

    };
}}