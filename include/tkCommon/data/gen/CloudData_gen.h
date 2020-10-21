// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class CloudData_gen : public SensorData
{
public:
    typedef std::string featureType_t;
    static const featureType_t FEATURES_NONE;
    static const featureType_t FEATURES_I;
    static const featureType_t FEATURES_CHANNEL;
    static const featureType_t FEATURES_R;
    static const featureType_t FEATURES_G;
    static const featureType_t FEATURES_B;
    static const featureType_t FEATURES_NX;
    static const featureType_t FEATURES_NY;
    static const featureType_t FEATURES_NZ;
    static const featureType_t FEATURES_CLASS;
    static const featureType_t FEATURES_DIST;
    static const featureType_t FEATURES_VELOCITY;
    static const featureType_t FEATURES_FALSE_DET;
    static const featureType_t FEATURES_RANGE_VAR;
    static const featureType_t FEATURES_VELOCITY_VAR;
    static const featureType_t FEATURES_ANGLE_VAR;
    static const featureType_t FEATURES_RCS;
    static const featureType_t FEATURES_PROBABILITY;
    static const featureType_t FEATURES_NEAR_SCAN;
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
        os<<"CloudData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
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

const CloudData_gen::featureType_t CloudData_gen::FEATURES_NONE = "f_none";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_I = "f_intensity";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_CHANNEL = "f_channel";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_R = "f_r";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_G = "f_g";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_B = "f_b";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_NX = "f_nx";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_NY = "f_ny";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_NZ = "f_nz";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_CLASS = "f_class";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_DIST = "f_dist";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_VELOCITY = "f_velocity";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_FALSE_DET = "f_false_det";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_RANGE_VAR = "f_range_var";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_VELOCITY_VAR = "f_velocity_var";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_ANGLE_VAR = "f_angle_var";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_RCS = "f_rcs";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_PROBABILITY = "f_probability";
const CloudData_gen::featureType_t CloudData_gen::FEATURES_NEAR_SCAN = "f_near_scan";


}}
