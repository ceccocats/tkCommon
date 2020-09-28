// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class GpsData_gen : public SensorData
{
public:
    double lat = 0;
    double lon = 0;
    double heigth = 0;
    int quality = 0;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
