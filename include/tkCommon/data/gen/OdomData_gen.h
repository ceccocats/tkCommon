// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace data {

class OdomData_gen : public SensorData
{
public:
    tk::math::Mat4f tf;
    
    void init() override
    {
        SensorData::init();
    }
    OdomData_gen& operator=(const OdomData_gen& s)
    {
        SensorData::operator=(s);
        tf = s.tf;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const OdomData_gen& s)
    {
        os<<"OdomData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	tf: "<<s.tf<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(2);
        structVars[0].set("header", header);
        structVars[1].set("tf", tf);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["tf"].get(tf);
        return true;
    }
};



}}
