// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class ImageData_gen : public SensorData
{
public:
    tk::math::Mat<uint8_t> data;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
