#pragma once

#include "tkCommon/common.h"
#include "tkCommon/math/MatIO.h"
#include <mutex>

namespace tk { namespace data {

    class sensorType{

    public:
        enum Value : uint8_t{
        NOT_SPEC    = 0,
        LIDAR       = 1,
        VEHICLE     = 2,
        GPS         = 3,
        CAMDATA     = 4,
        RADAR       = 5,
        LINES       = 6,
        PERCEPTION  = 7,
        IMU         = 8,
        GPSIMU      = 9,
        STEREO      = 10,
        CAN         = 11
        };

        /**
         * @brief   method for convert id to lane string name
         */
        std::string toString(){
            if(value == sensorType::NOT_SPEC)   return std::string{"not specified"};
            if(value == sensorType::LIDAR)      return std::string{"lidar"};
            if(value == sensorType::VEHICLE)    return std::string{"vehicle"};
            if(value == sensorType::GPS)        return std::string{"gps"};
            if(value == sensorType::CAMDATA)    return std::string{"camera"};
            if(value == sensorType::RADAR)      return std::string{"radar"};
            if(value == sensorType::PERCEPTION) return std::string{"perception"};
            if(value == sensorType::IMU)        return std::string{"imu"};
            if(value == sensorType::GPSIMU)     return std::string{"gps&imu"};
            return std::string{"type error"};
        }

        bool operator!=(sensorType::Value v) noexcept {
            return v != value;
        }

        bool operator==(sensorType::Value v) noexcept {
            return v == value;
        }

        bool operator!=(sensorType &s) noexcept {
            return s.value != value;
        }

        bool operator==(sensorType &s) noexcept {
            return s.value == value;
        }

        void operator=(sensorType::Value v) noexcept {
            value = v;
        }

    private:
        sensorType::Value value = sensorType::Value::NOT_SPEC;
    };


    /**
     * @brief Header data class.
     * Standard metadata for higher-level data class.
     */
    class HeaderData : public tk::math::MatDump {
    public:
        std::string         name   = "";                             /**< Name of the sensor. */
        tk::common::Tfpose  tf     = tk::common::Tfpose::Identity(); /**< TF in respect to back axel, @see tk::common::Tfpose. */
        sensorType          type;
        
        timeStamp_t         stamp     = 0; /**< Time stamp, expressed in microseconds. */
        int                 sensorID  = 0; /**< ID of the sensor. */
        int                 messageID = 0; /**< Incremental message counter. */

        float               fps = 0;

        void init() {
            this->stamp         = 0;
            this->tf            = tk::common::Tfpose::Identity();
            this->sensorID      = 0;
            this->messageID     = 0;
            this->type          = sensorType::NOT_SPEC;
            this->fps           = 0;
        }

        /**
         * @brief Overloading of operator = for class copy.
         *
         * @param s
         * @return
         */
        HeaderData &operator=(const HeaderData &s) {
            this->stamp         = s.stamp;
            this->sensorID      = s.sensorID;
            this->messageID     = s.messageID;
            this->type          = s.type;
            this->fps           = s.fps;

            // safe string update
            if(this->name != s.name) {
                tkWRN("Changing header name from " + name + " to " + s.name +
                       " if this message appers it is a problem\n");
                // this WARNING could be caused by the use of a tmpData in the sensor
                // if you are using tmpData please be sure to not rewrite the header name
                this->name = s.name;
            }
            // safe TF update
            tf = s.tf;

            return *this;
        }

        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(3);
            structVars[0].set("stamp", stamp);
            structVars[1].set("tf", tf.matrix());
            structVars[2].set("name", this->name);
            return var.setStruct(name, structVars);
        }
        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            var["stamp"].get(stamp);
            var["tf"].get(tf.matrix());
            var["name"].get(this->name);
            return true;
        }
    };
}}
