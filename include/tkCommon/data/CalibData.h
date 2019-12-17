#include "tkCommon/common.h"

namespace tk{ namespace data{

    class CalibData{
    
    public:

        /**
         * @brief Intrinsic calibration matrix (3x3)
         */
        float k[9];
        /**
         * @brief Radial distortion params (1x5)
         */
        float d[5];
        /**
         * @brief Inverse rotation matrix (3x3)
         */
        float r[9];

        /**
         * @brief initialization with default values
         */
        void init(){
			memset(k, 0, 9 * sizeof(float));
			k[0] = 1; k[4] = 1; k[8] = 1;
			memset(r, 0, 9 * sizeof(float));
			r[0] = 1; r[4] = 1; r[8] = 1;
			memset(d, 0, 5 * sizeof(float));
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

    };

}}