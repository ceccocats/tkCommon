#pragma once
#include "tkCommon/math/MatIO.h"
#include "tkCommon/data/HeaderData.h"

namespace tk { namespace data {

    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
    class SensorData : tk::math::MatDump {
    public:
        HeaderData  header;                 /**< Header, @see HeaderData */

        /**
         * @brief Initialization method.
         * Must be implemented by child classes, and will handle the allocation of member variables, if any.
         */
        virtual void init() {
            header.init();
        }

        /**
         * @brief Release method.
         * Must be implemented by child classes, and will handle the deallocation of member variables, if any,
         */
        virtual void release() { clsErr("release method not implemented"); tkFATAL("abort"); };

        virtual bool checkDimension(SensorData *s) { clsErr("check dimension method not implemented"); tkFATAL("abort"); };

        /**
         * @brief Overloading of operator =
         * Copy only the header.
         *
         * @param s
         * @return
         */
        SensorData& operator=(const SensorData &s) {
            tkASSERT(checkDimension((SensorData*)&s));
            this->header        = s.header;
            return *this;
        }

        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            structVars[0].set("stamp", header.stamp);
            structVars[1].set("tf", header.tf.matrix());
            return var.setStruct(name, structVars);
        }
        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;
            var["stamp"].get(header.stamp);
            var["tf"].get(header.tf.matrix());
            return true;
        }
    };
}}
