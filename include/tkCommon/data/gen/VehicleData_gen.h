// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class VehicleData_gen : public SensorData
{
public:
    double speed;
    double yawRate;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
