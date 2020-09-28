// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class ImuData_gen : public SensorData
{
public:
    tk::common::Vector3<double> acc;
    tk::common::Vector3<double> angleRate;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
