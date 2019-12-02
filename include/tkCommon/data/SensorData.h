#pragma once

#include "tkCommon/data/HeaderData.h"

namespace tk { namespace data {

    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
    class SensorData {
    public:
        HeaderData  header;                 /**< Header, @see HeaderData */
        bool        isInitilized = false;   /**< */

        /**
         * @brief Constructor.
         */
        SensorData() = default;

        /**
         * @brief Destructor.
         */
        virtual ~SensorData() = default;

        /**
         * @brief Initialization method.
         * Must be implemented by child classes, and will handle the allocation of member variables, if any.
         */
        virtual void init() = 0;

        /**
         * @brief Release method.
         * Must be implemented by child classes, and will handle the deallocation of member variables, if any,
         */
        virtual void release() = 0;

        /**
         * @brief Overloading of operator =
         * Copy only the header.
         *
         * @param s
         * @return
         */
        SensorData& operator=(const SensorData& s){
            this->header        = s.header;

            return *this;
        }
    };
}}
