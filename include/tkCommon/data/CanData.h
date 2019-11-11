#pragma once
#include <tkCommon/common.h>
#include <linux/can.h>
#include <linux/can/raw.h>

namespace tk { namespace data {
    struct CanData_t {
        tk::data::DataHeader_t header;

        struct can_frame frame;     // socketCAN frame
        timeStamp_t stamp;          // timestap of message when read

        // return integer ID
        int     id()     { return frame.can_id; }
       
        // return data as a uint64_t pointer, useful for DBC encoding
        uint64_t *data() { return reinterpret_cast<uint64_t*>(frame.data); }
    };
}}
