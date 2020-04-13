#pragma once

#include "tkCommon/common.h"

namespace tk { namespace data {

    class sensorName{

    public:
        enum Value : uint8_t{
        NOT_SPEC    = 0,
        LIDAR       = 1,
        VEHICLE     = 2,
        GPS         = 3,
        CAMDATA     = 4,
        RADAR       = 5,
        LINES       = 6
        };

        /**
         * @brief   method for convert id to lane string name
         */
        std::string toString(){
            if(value == sensorName::NOT_SPEC)   return std::string{"not specified"};
            if(value == sensorName::LIDAR)      return std::string{"lidar"};
            if(value == sensorName::VEHICLE)    return std::string{"vehicle"};
            if(value == sensorName::GPS)        return std::string{"gps"};
            if(value == sensorName::CAMDATA)    return std::string{"camera"};
            if(value == sensorName::RADAR)      return std::string{"radar"};
            return std::string{"type error"};
        }

        bool operator!=(sensorName::Value v) noexcept {
            return v != value;
        }

        bool operator==(sensorName::Value v) noexcept {
            return v == value;
        }

        bool operator!=(sensorName &s) noexcept {
            return s.value != value;
        }

        bool operator==(sensorName &s) noexcept {
            return s.value == value;
        }

        void operator=(sensorName::Value v) noexcept {
            value = v;
        }

    private:
        sensorName::Value value;
    };


    /**
     * @brief Header data class.
     * Standard metadata for higher-level data class.
     */
    class HeaderData {
    public:
        sensorName          sensor;
        timeStamp_t         stamp = 0;      /**< Time stamp, expressed in millisecond. */
        std::string         name;           /**< Name of the sensor. */
        tk::common::Tfpose  tf;             /**< TF in respect to back axel, @see tk::common::Tfpose. */
        int                 sensorID;       /**< ID of the sensor. */
        int                 messageID;      /**< Incremental message counter. */

        void init() {
            this->stamp         = 0;
            this->tf            = tk::common::Tfpose::Identity();
            this->sensorID      = 0;
            this->messageID     = 0;
            this->sensor        = sensorName::NOT_SPEC;
        }

        /**
         * @brief Overloading of operator = for class copy.
         *
         * @param s
         * @return
         */
        HeaderData &operator=(const HeaderData &s) {
            this->stamp         = s.stamp;
            this->tf            = s.tf;
            this->sensorID      = s.sensorID;
            this->messageID     = s.messageID;
            this->name          = s.name;
        }
    };
}}
