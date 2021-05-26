// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

class CloudData_gen : public SensorData
{
public:
    typedef std::string featureType_t;
    static const DataType type;
    static const CloudData_gen::featureType_t FEATURES_NONE;
    static const CloudData_gen::featureType_t FEATURES_I;
    static const CloudData_gen::featureType_t FEATURES_NOISE;
    static const CloudData_gen::featureType_t FEATURES_SIGNAL_NOISE;
    static const CloudData_gen::featureType_t FEATURES_CHANNEL;
    static const CloudData_gen::featureType_t FEATURES_CHANNEL_H;
    static const CloudData_gen::featureType_t FEATURES_R;
    static const CloudData_gen::featureType_t FEATURES_G;
    static const CloudData_gen::featureType_t FEATURES_B;
    static const CloudData_gen::featureType_t FEATURES_NX;
    static const CloudData_gen::featureType_t FEATURES_NY;
    static const CloudData_gen::featureType_t FEATURES_NZ;
    static const CloudData_gen::featureType_t FEATURES_CLASS;
    static const CloudData_gen::featureType_t FEATURES_DIST;
    static const CloudData_gen::featureType_t FEATURES_VELOCITY;
    static const CloudData_gen::featureType_t FEATURES_FALSE_DET;
    static const CloudData_gen::featureType_t FEATURES_RANGE_VAR;
    static const CloudData_gen::featureType_t FEATURES_VELOCITY_VAR;
    static const CloudData_gen::featureType_t FEATURES_ANGLE_VAR;
    static const CloudData_gen::featureType_t FEATURES_RCS;
    static const CloudData_gen::featureType_t FEATURES_PROBABILITY;
    static const CloudData_gen::featureType_t FEATURES_NEAR_SCAN;
    static const CloudData_gen::featureType_t FEATURES_GROUND;
    static const CloudData_gen::featureType_t FEATURES_TIME;
    static const CloudData_gen::featureType_t FEATURES_REFLECTOR;
    tk::math::Mat<float> points;
    tk::math::Mat<float> ranges;
    tk::common::Map<tk::math::Vec<float>> features;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    CloudData_gen& operator=(const CloudData_gen& s)
    {
        SensorData::operator=(s);
        points = s.points;
        ranges = s.ranges;
        features = s.features;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, CloudData_gen& s)
    {
        os<<"CloudData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	points: "<<s.points<<std::endl;
        os<<"	ranges: "<<s.ranges<<std::endl;
        os<<"	features: "<<s.features<<std::endl;
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
