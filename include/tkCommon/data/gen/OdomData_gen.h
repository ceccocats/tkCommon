// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace data {

class OdomData_gen : public SensorData
{
public:
    float x;
    float y;
    float yaw;
    float speed;
    
    void init() override
    {
        SensorData::init();
    }
    OdomData_gen& operator=(const OdomData_gen& s)
    {
        SensorData::operator=(s);
        x = s.x;
        y = s.y;
        yaw = s.yaw;
        speed = s.speed;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, OdomData_gen& s)
    {
        os<<"OdomData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	x: "<<s.x<<std::endl;
        os<<"	y: "<<s.y<<std::endl;
        os<<"	yaw: "<<s.yaw<<std::endl;
        os<<"	speed: "<<s.speed<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0].set("header", header);
        structVars[1].set("x", x);
        structVars[2].set("y", y);
        structVars[3].set("yaw", yaw);
        structVars[4].set("speed", speed);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["x"].get(x);
        var["y"].get(y);
        var["yaw"].get(yaw);
        var["speed"].get(speed);
        return true;
    }
};


}}
