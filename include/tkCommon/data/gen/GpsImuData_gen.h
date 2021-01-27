// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/ImuData.h"

#include "tkCommon/data/GpsData.h"


namespace tk { namespace data {

class GpsImuData_gen : public SensorData
{
public:
    tk::data::GpsData gps;
    tk::data::ImuData imu;
    tk::math::Vec3<double> vel;
    
    void init() override
    {
        SensorData::init();
        gps.init();;
        imu.init();;
    }
    GpsImuData_gen& operator=(const GpsImuData_gen& s)
    {
        SensorData::operator=(s);
        gps = s.gps;
        imu = s.imu;
        vel = s.vel;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, GpsImuData_gen& s)
    {
        os<<"GpsImuData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	gps: "<<s.gps<<std::endl;
        os<<"	imu: "<<s.imu<<std::endl;
        os<<"	vel: "<<s.vel<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("gps", gps);
        structVars[2].set("imu", imu);
        structVars[3].set("vel", vel);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["gps"].get(gps);
        var["imu"].get(imu);
        var["vel"].get(vel);
        return true;
    }
};


}}
