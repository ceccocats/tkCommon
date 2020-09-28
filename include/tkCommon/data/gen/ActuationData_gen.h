// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class ActuationData_gen : public SensorData
{
public:
    double steerAngle;
    double accel;
    double speed;
    
    void init() override
    {
        SensorData::init();
        0;
        0;
        0;
    }
};

}}
