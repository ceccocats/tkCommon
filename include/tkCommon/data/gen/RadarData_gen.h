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
        near.init();
        far.init();
    }
    RadarData_gen& operator=(const RadarData_gen& s)
    {
        SensorData::operator=(s);
        near = s.near;
        far = s.far;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, RadarData_gen& s)
    {
        os<<"RadarData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	near: "<<s.near<<std::endl;
        os<<"	far: "<<s.far<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("near", near);
        structVars[2].set("far", far);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["near"].get(near);
        var["far"].get(far);
        return true;
    }
};



}}
