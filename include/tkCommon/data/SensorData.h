#pragma once

#include "tkCommon/gui/Drawable.h"
#include "tkCommon/data/HeaderData.h"

#include "tkCommon/gui/Viewer.h"

namespace tk { namespace data {

    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
	class SensorData : public tk::gui::Drawable {
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
        virtual void release() = 0;

        virtual bool checkDimension(SensorData *s) = 0;

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
    };
}}
