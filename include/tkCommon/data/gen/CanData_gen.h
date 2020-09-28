// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include <linux/can.h>
#include <linux/can/raw.h>

namespace tk { namespace data {

class CanData_gen : public SensorData
{
public:
    struct can_frame frame;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
