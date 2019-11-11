#pragma once

#include "tkCommon/common.h"
namespace tk { namespace data {

    struct DataHeader_t {
        timeStamp_t stamp = 0;
        tk::common::Tfpose tf = tk::common::Tfpose::Identity();
        int sensorID = 0;
        int messageID = 0;

        /**
         * @brief Overloading for struct copy
         *
         */
        DataHeader_t &operator=(const DataHeader_t &s) {
            this->stamp = s.stamp;
            this->tf = s.tf;
            this->sensorID = s.sensorID;
            this->messageID = s.messageID;
        }
    };

}}