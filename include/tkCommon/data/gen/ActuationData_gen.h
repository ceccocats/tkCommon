// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class ActuationData_gen : public SensorData
{
public:
    double steerAngle;
    double accel;
    double speed;
    
    void init() override
    {
        SensorData::init();
        steerAngle = 0;
        accel = 0;
        speed = 0;
    }
    ActuationData_gen& operator=(const ActuationData_gen& s)
    {
        SensorData::operator=(s);
        steerAngle = s.steerAngle;
        accel = s.accel;
        speed = s.speed;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const ActuationData_gen& s)
    {
        os<<"ActuationData_gen"<<std::endl;
        os<<"	header.stamp:"<<s.header.stamp<<std::endl;
        os<<"	steerAngle: "<<s.steerAngle<<std::endl;
        os<<"	accel: "<<s.accel<<std::endl;
        os<<"	speed: "<<s.speed<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("steerAngle", steerAngle);
        structVars[2].set("accel", accel);
        structVars[3].set("speed", speed);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["steerAngle"].get(steerAngle);
        var["accel"].get(accel);
        var["speed"].get(speed);
        return true;
    }
};



}}
