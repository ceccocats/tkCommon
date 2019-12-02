#pragma once

#include "tkCommon/common.h"

namespace tk { namespace data {

    /**
     * @brief Header data class.
     * Standard metadata for higher-level data class.
     */
    class HeaderData {
    public:
        timeStamp_t         stamp;      /**< Time stamp, expressed in millisecond. */
        std::string         name;       /**< Name of the sensor. */
        tk::common::Tfpose  tf;         /**< TF in respect to back axel, @see tk::common::Tfpose. */
        int                 sensorID;   /**< ID of the sensor. */
        int                 messageID;  /**< Incremental message counter. */

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
    };
}}
