#pragma once

#include "tkCommon/common.h"
#include "tkCommon/math/MatIO.h"
#include <mutex>

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
        LINES       = 6,
        PERCEPTION  = 7,
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
            if(value == sensorName::PERCEPTION) return std::string{"perception"};
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
        sensorName::Value value = sensorName::Value::NOT_SPEC;
    };


    /**
     * @brief Header data class.
     * Standard metadata for higher-level data class.
     */
    class HeaderData : public tk::math::MatDump {
    public:
        std::string         name   = "";                             /**< Name of the sensor. */
        tk::common::Tfpose  tf     = tk::common::Tfpose::Identity(); /**< TF in respect to back axel, @see tk::common::Tfpose. */
        sensorName          sensor;
        
        timeStamp_t         stamp     = 0; /**< Time stamp, expressed in millisecond. */
        int                 sensorID  = 0; /**< ID of the sensor. */
        int                 messageID = 0; /**< Incremental message counter. */

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
            this->sensorID      = s.sensorID;
            this->messageID     = s.messageID;

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
            std::vector<tk::math::MatIO::var_t> structVars(2);
            structVars[0].set("stamp", stamp);
            structVars[1].set("tf", tf.matrix());
            return var.setStruct(name, structVars);
        }
        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            var["stamp"].get(stamp);
            var["tf"].get(tf.matrix());
            return true;
        }
    };
}}
