#pragma once

#include "tkCommon/common.h"
#include "tkCommon/data/HeaderData.h"

namespace tk { namespace data {

    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
    class SensorData {
    public:
        /**
         * @brief Constructor.
         */
        SensorData();

        /**
         * @brief Destructor.
         */
        virtual ~SensorData() = default;

        /**
         * @brief Initialization method.
         */
        virtual void init() = 0;

        /**
         * @brief
         */
        virtual void release() = 0;

        /**
         * @brief Overloading of operator =
         * This method must be overridden by child class.
         *
         * @param s
         * @return
         */
        virtual SensorData &operator= (const SensorData& s) = 0;

        HeaderData  header;         /**< Header, @see HeaderData */
        bool        isInitilized;   /**< */
    };

    SensorData::SensorData() {
        this->isInitilized = false;
    }
}}
