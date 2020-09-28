// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/gen/CloudData_gen.h"


namespace tk { namespace data {

class RadarData_gen : public SensorData
{
public:
    tk::data::CloudData_gen near;
    tk::data::CloudData_gen far;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
