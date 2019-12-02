#pragma once

#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/CloudData.h"
#include <vector>

namespace tk { namespace data {
    static const int RADAR_MAX = 8;
    static const int RADAR_MAX_PONINTS = 10000;

    /**
     * @brief Radar data class.
     * Contains all the information generated from a generic Radar sensor.
     */
    class RadarData : public SensorData {
    public:
        int nRadar;     /**< Number of radar  */
        std::vector<tk::data::CloudData>  near_data;    /**< Vector of @see tk::data::CloudData */
        std::vector<tk::data::CloudData>  far_data;     /**< Vector of @see tk::data::CloudData */

        /**
         * @brief Costructor.
         */
        RadarData();

        /**
         * @brief Desctructor.
         */
        ~RadarData() override = default;

        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic vector of @see tk::data::CloudData at maximum size.
         */
        void init() override;

        /**
         * @brief Release method.
         * Handles the deallocation of of dynamic vector of @see tk::data::CloudData.
         */
        void release() override;

        /**
         * @brief Overloading of operator = for class copy.
         * @param s
         * @return
         */
        RadarData& operator=(const RadarData& s);
    };
}}