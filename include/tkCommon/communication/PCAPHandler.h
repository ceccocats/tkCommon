#pragma once

#include "tkCommon/common.h"
#include "tkCommon/utils.h"

namespace tk { namespace communication {
    class PCAPHandler {
    public:
        PCAPHandler();
        ~PCAPHandler();

        bool initReplay(const std::string fileName, const std::string filter);
        bool initRecord(const std::string fileName, const std::string filter);

        void record();
        int getPacket(u_int8_t& buffer, timeStamp_t& stamp);

        bool close();
    private:
    };
}}