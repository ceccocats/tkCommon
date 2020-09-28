// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class CalibData_gen : public SensorData
{
public:
    tk::math::Mat<float> K;
    tk::math::Mat<float> D;
    tk::math::Mat<float> R;
    
    void init() override
    {
        SensorData::init();
        K.resize(3,3);
        D.resize(1,5);
        R.resize(3,3);
    }
};

}}
