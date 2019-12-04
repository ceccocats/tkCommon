#include "tkCommon/common.h"

namespace tk{ namespace data{

    class CalibData{
    
    public:

        /**
         * @brief Intrinsic calibration matrix (3x3)
         */
        double k[9];
        /**
         * @brief Radial distortion params (1x5)
         */
        double d[5];
        /**
         * @brief Inverse rotation matrix (3x3)
         */
        double r[9];

        /**
         * @brief       Method that load a calibartion file
         * 
         * @param conf  yaml node config
         * 
         *              K           |D          |R      |
         *              :----------:|:---------:|:---------:|
         *              double(9)   |double(5)  |double(9)  |
         * @return status
         */
        bool load(const YAML::Node conf){

            memset(k,0.0,9*sizeof(double));
            memset(d,0.0,5*sizeof(double));
            memset(r,0.0,9*sizeof(double));

            std::vector<double> k_tmp = getYAMLconf<std::vector<double>>(conf, "K", std::vector<double>(9,0));
            tkASSERT(k_tmp.size() != 9)
            for(int i = 0; i < 9; i++)
                k[i] = k_tmp[i];

            std::vector<double> d_tmp = getYAMLconf<std::vector<double>>(conf, "D", std::vector<double>(5,0));
            tkASSERT(k_tmp.size() != 5)
            for(int i = 0; i < 5; i++)
                d[i] = d_tmp[i];

            std::vector<double> r_tmp = getYAMLconf<std::vector<double>>(conf, "R", std::vector<double>(9,0));
            tkASSERT(r_tmp.size() != 9)
            for(int i = 0; i < 9; i++)
                r[i] = r_tmp[i];

            return true;
        }

    };

}}