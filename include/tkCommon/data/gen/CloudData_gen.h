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
    CloudData_gen& operator=(const CloudData_gen& s)
    {
        SensorData::operator=(s);
        points = s.points;
        ranges = s.ranges;
        features = s.features;
        features_map = s.features_map;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const CloudData_gen& s)
    {
        os<<"CloudData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"points: "<<s.points<<std::endl;
        os<<"ranges: "<<s.ranges<<std::endl;
        os<<"features: "<<s.features<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("points", points);
        structVars[2].set("ranges", ranges);
        structVars[3].set("features", features);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["points"].get(points);
        var["ranges"].get(ranges);
        var["features"].get(features);
        return true;
    }
};

}}
