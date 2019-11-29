#pragma once

#include "tkCommon/common.h"

namespace tk { namespace data {

    /**
     * @brief Header data class.
     * Standard metadata for higher-level data class.
     */
    class HeaderData {
    public:
        /**
         * @brief Constructor.
         */
        HeaderData();

        /**
         * @brief Destructor.
         */
        ~HeaderData() = default;

        /**
         * @brief Overloading of operator = for class copy.
         *
         * @param s
         * @return
         */
        HeaderData &operator=(const HeaderData &s);

        timeStamp_t         stamp;      /**< Time stamp, expressed in millisecond. */
        std::string         name;       /**< Name of the sensor. */
        tk::common::Tfpose  tf;         /**< TF in respect to back axel, @see tk::common::Tfpose. */
        int                 sensorID;   /**< ID of the sensor. */
        int                 messageID;  /**< Incremental message counter. */
    };

    HeaderData::HeaderData() {
        this->stamp         = 0;
        this->tf            = tk::common::Tfpose::Identity();
        this->sensorID      = 0;
        this->messageID     = 0;
        this->name          = "sensor";
    }

    HeaderData&
    HeaderData::operator=(const HeaderData &s) {
        this->stamp         = s.stamp;
        this->tf            = s.tf;
        this->sensorID      = s.sensorID;
        this->messageID     = s.messageID;
    }

}}
