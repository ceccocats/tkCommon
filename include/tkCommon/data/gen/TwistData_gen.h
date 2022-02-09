// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

class TwistData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec3<double> linear;
    tk::math::Vec3<double> angular;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    TwistData_gen& operator=(const TwistData_gen& s)
    {
        SensorData::operator=(s);
        linear = s.linear;
        angular = s.angular;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, TwistData_gen& s)
    {
        os<<"TwistData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	linear: "<<s.linear<<std::endl;
        os<<"	angular: "<<s.angular<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("linear", linear);
        structVars[2].set("angular", angular);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["linear"].get(linear);
        var["angular"].get(angular);
        return true;
    }
};


}}
