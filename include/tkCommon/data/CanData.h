#pragma once
#include <tkCommon/common.h>
#include <linux/can.h>
#include <linux/can/raw.h>

namespace tk { namespace data {
    struct CanData_t {
        struct can_frame frame;     // socketCAN frame
        timeStamp_t stamp;          // timestap of message when read

        // return integer ID
        int     id()     { return frame.can_id; }
       
        // return data as a uint64_t pointer, useful for DBC encoding
        uint64_t *data() { return reinterpret_cast<uint64_t*>(frame.data); }

        friend std::ostream& operator<<(std::ostream& os, const CanData_t& m) {
            os << std::setprecision(10) << m.stamp<< "\t"<<std::hex<<m.frame.can_id<<std::dec;
            os << "\t"<<std::hex;
            for(int i=0; i<m.frame.can_dlc; i++) {
                os << std::setfill('0') << std::setw(2) << int(m.frame.data[i]);
            }
            os << std::dec;
            return os;
        }
    };
}}
