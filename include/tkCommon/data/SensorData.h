#pragma once
#include "tkCommon/gui/utils/Drawable.h"
#include "tkCommon/math/MatIO.h"
#include "tkCommon/data/HeaderData.h"

#include "tkCommon/gui/Viewer.h"

namespace tk { namespace data {

    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
	class SensorData : public tk::gui::Drawable, public tk::math::MatDump {
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
        friend std::ostream& operator<<(std::ostream& os, const SensorData& s)
        {
            os<<"header.stamp: "<<s.header.stamp;
            return os;
        }

        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            return header.toVar(name, var);
        }
        bool fromVar(tk::math::MatIO::var_t &var) {
            return header.fromVar(var);
        }
    };
}}
