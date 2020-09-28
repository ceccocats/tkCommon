// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class CloudData_gen : public SensorData
{
public:
    typedef std::string featureType_t;
    const featureType_t FEATURES_NONE = "f_none";
    const featureType_t FEATURES_I = "f_intensity";
    tk::math::Mat<float> points;
    tk::math::Mat<float> ranges;
    tk::math::Mat<float> features;
    std::map<featureType_t, int> features_map;
    
    void init() override
    {
        SensorData::init();
    }
};

}}
