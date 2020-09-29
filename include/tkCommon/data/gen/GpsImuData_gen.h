// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/gen/ImuData_gen.h"

#include "tkCommon/data/gen/GpsData_gen.h"


namespace tk { namespace data {

class GpsImuData_gen : public SensorData
{
public:
    tk::data::GpsData_gen gps;
    tk::data::ImuData_gen imu;
    
    void init() override
    {
        SensorData::init();
    }
    GpsImuData_gen& operator=(const GpsImuData_gen& s)
    {
        SensorData::operator=(s);
        gps = s.gps;
        imu = s.imu;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const GpsImuData_gen& s)
    {
        os<<"GpsImuData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"gps: "<<s.gps<<std::endl;
        os<<"imu: "<<s.imu<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("gps", gps);
        structVars[2].set("imu", imu);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["gps"].get(gps);
        var["imu"].get(imu);
        return true;
    }
};

}}
