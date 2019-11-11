#pragma once

#include "tkCommon/common.h"

struct DataHeader_t {
    timeStamp_t         stamp;
    tk::common::Tfpose  tf;
    int                 sensorID;
    int                 messageID;

    /**
     * @brief Overloading for struct copy
     *
     */
    DataHeader_t& operator=(const DataHeader_t& s){
        this->stamp     = s.stamp;
        this->tf        = s.tf;
        this->sensorID  = s.sensorID;
        this->messageID = s.messageID;
    }
};