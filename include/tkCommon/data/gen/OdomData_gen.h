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
    tk::math::Vec4<double> angle;
    tk::math::Vec3<double> linear_velocity;
    tk::math::Vec3<double> angular_velocity;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    OdomData_gen& operator=(const OdomData_gen& s)
    {
        SensorData::operator=(s);
        pose = s.pose;
        angle = s.angle;
        linear_velocity = s.linear_velocity;
        angular_velocity = s.angular_velocity;
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
        os<<"	linear_velocity: "<<s.linear_velocity<<std::endl;
        os<<"	angular_velocity: "<<s.angular_velocity<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0].set("header", header);
        structVars[1].set("pose", pose);
        structVars[2].set("angle", angle);
        structVars[3].set("linear_velocity", linear_velocity);
        structVars[4].set("angular_velocity", angular_velocity);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["pose"].get(pose);
        var["angle"].get(angle);
        var["linear_velocity"].get(linear_velocity);
        var["angular_velocity"].get(angular_velocity);
        return true;
    }
};


}}
