// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class GPSData : public SensorData
{
public:
    double lat;
    double lon;
    double heigth;
    int quality;
    
    void init() override
    {
        SensorData::init();
        lat = 0;
        lon = 0;
        heigth = 0;
        quality = 0;
    }
    GPSData& operator=(const GPSData& s)
    {
        SensorData::operator=(s);
        lat = s.lat;
        lon = s.lon;
        heigth = s.heigth;
        quality = s.quality;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const GPSData& s)
    {
        os<<"GPSData:"<<std::endl;
        os<<"lat: "<<s.lat<<std::endl;
        os<<"lon: "<<s.lon<<std::endl;
        os<<"heigth: "<<s.heigth<<std::endl;
        os<<"quality: "<<s.quality<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        tk::math::MatIO::var_t hvar;
        tk::data::SensorData::toVar("header", hvar);
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0] = hvar;
        structVars[1].set("lat", lat);
        structVars[2].set("lon", lon);
        structVars[3].set("heigth", heigth);
        structVars[4].set("quality", quality);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        tk::data::SensorData::fromVar(var["header"]);
        var["lat"].get(lat);
        var["lon"].get(lon);
        var["heigth"].get(heigth);
        var["quality"].get(quality);
        return true;
    }
};

}}
