// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"

namespace tk { namespace data {

class ImuData_gen : public SensorData
{
public:
    tk::math::Vec3<double> acc;
    tk::math::Vec3<double> angleRate;
    
    void init() override
    {
        SensorData::init();
    }
    ImuData_gen& operator=(const ImuData_gen& s)
    {
        SensorData::operator=(s);
        acc = s.acc;
        angleRate = s.angleRate;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const ImuData_gen& s)
    {
        os<<"ImuData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"acc: "<<s.acc<<std::endl;
        os<<"angleRate: "<<s.angleRate<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("acc", acc);
        structVars[2].set("angleRate", angleRate);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["acc"].get(acc);
        var["angleRate"].get(angleRate);
        return true;
    }
};

}}
