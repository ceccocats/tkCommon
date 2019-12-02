#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {
    static const int CLOUD_MAX_POINTS       = 200000;


    /**
     * @brief Cloud data class.
     * Contains all the information of a generic point cloud. Both in cartesian and polar coordinates.
     */
    class CloudData : public SensorData {
    public:
        int                 nPoints;    /**< Number of points */
        Eigen::MatrixXf     points;     /**< Points in cartesian coordinates */
        Eigen::MatrixXf     ranges;     /**< Points in polar coordinates */



        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic eigen matrices to the maximum size.
         */
        void init() {
            init(CLOUD_MAX_POINTS);
        }

        /**
         * @brief Initialization method.
         * Handles the allocation of dynamic eigen matrices to the size passed as parameter.
         * @param size
         */
        void init(int size) {
            SensorData::init();
            tkASSERT(size <= CLOUD_MAX_POINTS);

            this->points.resize(4, size);
            this->ranges.resize(2, size);
        }

        /**
         * @brief Release method.
         * Deallocate eigen matrix.
         */
        void release() {
            this->nPoints = 0;
            this->points.resize(0, 0);
            this->ranges.resize(0, 0);
        }

        bool checkDimension(SensorData *s) {
            auto *source = dynamic_cast<CloudData*>(s);
            if (source->points.rows() > this->points.rows() ||
                source->ranges.rows() > this->ranges.rows()
                ) {
                return false;
            } else {
                return true;
            }
        }

        /**
         * @brief Overloading of operator = for class copy.
         * @param s
         * @return
         */
         CloudData& operator=(const CloudData &s) {
            SensorData::operator=(s);

            this->nPoints       = s.nPoints;
            std::memcpy(points.data(),   s.points.data(),   this->nPoints * 4 * sizeof(float));
            std::memcpy(ranges.data(),   s.ranges.data(),   this->nPoints * 2 * sizeof(float));

            return *this;
         }
    };
}}