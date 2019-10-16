#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    struct SensorEthData_t {
        const static int MAX_SIZE = 8192;
        timeStamp_t stamp;          // timestap of message when read
        uint8_t     buffer[MAX_SIZE];
        int         size;
    };

}}
