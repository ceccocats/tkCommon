// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"

namespace tk { namespace data {

class OdomData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec3<double> pose;
    tk::math::Vec3<double> angle;
    double speed;
    
    void init() override
    {
        SensorData::init();
        speed = 0;
    }
    OdomData_gen& operator=(const OdomData_gen& s)
    {
        SensorData::operator=(s);
        pose = s.pose;
        angle = s.angle;
        speed = s.speed;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, OdomData_gen& s)
    {
        os<<"OdomData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	pose: "<<s.pose<<std::endl;
        os<<"	angle: "<<s.angle<<std::endl;
        os<<"	speed: "<<s.speed<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("pose", pose);
        structVars[2].set("angle", angle);
        structVars[3].set("speed", speed);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["pose"].get(pose);
        var["angle"].get(angle);
        var["speed"].get(speed);
        return true;
    }
};


}}
