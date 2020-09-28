// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/gen/ImuData_gen.h"

#include "tkCommon/data/gen/GpsData_gen.h"


namespace tk { namespace data {

class GpsImuData_gen : public SensorData
{
public:
    tk::data::GpsData_gen gps;
    tk::data::ImuData_gen imu;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
