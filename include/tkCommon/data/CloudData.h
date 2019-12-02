#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {
    static const int CLOUD_MAX_POINTS       = 200000;
    static const int CLOUD_MAX_FEATURES     = 16;

    /**
     *
     */
    enum CloudFeatureType {
        R              = 0,
        G              = 1,
        B              = 2,
        I              = 3,
        CLASS          = 4,
        VELOCITY       = 5,
        FALSE_DET      = 6,
        RANGE_VAR      = 7,
        VELOCITY_VAR   = 8,
        ANGLE_VAR      = 9,
        RCS            = 10,
        ID             = 11
    };

    /**
     * @brief Cloud data class.
     * Contains all the information of a generic point cloud. Both in cartesian and polar coordinates.
     */
    class CloudData : public SensorData {
    public:
        int                 nPoints;    /**< Number of points */
        Eigen::MatrixXf     points;     /**< Points in cartesian coordinates */
        Eigen::MatrixXf     ranges;     /**< Points in polar coordinates */
        Eigen::MatrixXf     features;   /**< Feature of the cloud */

        /**
         * @brief Constructor.
         */
        CloudData();

        /**
         * @brief Destructor.
         */
        ~CloudData();

        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic eigen matrices to the maximum size.
         */
        void init() override;

        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic eigen matrices to the size passed as parameter.
         * @param size
         */
        void init(int size);

        /**
         * @brief Release method.
         * Deallocate eigen matrix.
         */
        void release() override;

        /**
         * @brief Overloading of operator = for class copy.
         * @param s
         * @return
         */
         CloudData& operator=(const CloudData& s);
    };
}}