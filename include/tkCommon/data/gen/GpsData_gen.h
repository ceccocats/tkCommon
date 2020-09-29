// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class GpsData_gen : public SensorData
{
public:
    double lat = 0;
    double lon = 0;
    double heigth = 0;
    int quality = 0;
    
    void init() override
    {
        SensorData::init();
    }
    GpsData_gen& operator=(const GpsData_gen& s)
    {
        SensorData::operator=(s);
        lat = s.lat;
        lon = s.lon;
        heigth = s.heigth;
        quality = s.quality;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const GpsData_gen& s)
    {
        os<<"GpsData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"lat: "<<s.lat<<std::endl;
        os<<"lon: "<<s.lon<<std::endl;
        os<<"heigth: "<<s.heigth<<std::endl;
        os<<"quality: "<<s.quality<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0].set("header", header);
        structVars[1].set("lat", lat);
        structVars[2].set("lon", lon);
        structVars[3].set("heigth", heigth);
        structVars[4].set("quality", quality);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["lat"].get(lat);
        var["lon"].get(lon);
        var["heigth"].get(heigth);
        var["quality"].get(quality);
        return true;
    }
};

}}
