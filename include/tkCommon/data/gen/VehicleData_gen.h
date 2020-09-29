// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class VehicleData_gen : public SensorData
{
public:
    double speed;
    double yawRate;
    
    void init() override
    {
        SensorData::init();
    }
    VehicleData_gen& operator=(const VehicleData_gen& s)
    {
        SensorData::operator=(s);
        speed = s.speed;
        yawRate = s.yawRate;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const VehicleData_gen& s)
    {
        os<<"VehicleData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"speed: "<<s.speed<<std::endl;
        os<<"yawRate: "<<s.yawRate<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("speed", speed);
        structVars[2].set("yawRate", yawRate);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["speed"].get(speed);
        var["yawRate"].get(yawRate);
        return true;
    }
};

}}
