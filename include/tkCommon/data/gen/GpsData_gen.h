// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace data {

class GpsData_gen : public SensorData
{
public:
    static const DataType type;
    timeStamp_t utcStamp;
    double lat;
    double lon;
    double heigth;
    double speed_hor;
    double speed_ver;
    double heading;
    int quality;
    int sats;
    int age;
    tk::math::Mat3d cov;
    double pressure;
    double temp;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
        utcStamp = 0;
        lat = 0;
        lon = 0;
        heigth = 0;
        speed_hor = 0;
        speed_ver = 0;
        heading = 0;
        quality = 0;
        sats = 0;
        age = 0;
        pressure = 0;
        temp = 0;
    }
    GpsData_gen& operator=(const GpsData_gen& s)
    {
        SensorData::operator=(s);
        utcStamp = s.utcStamp;
        lat = s.lat;
        lon = s.lon;
        heigth = s.heigth;
        speed_hor = s.speed_hor;
        speed_ver = s.speed_ver;
        heading = s.heading;
        quality = s.quality;
        sats = s.sats;
        age = s.age;
        cov = s.cov;
        pressure = s.pressure;
        temp = s.temp;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, GpsData_gen& s)
    {
        os<<"GpsData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	utcStamp: "<<s.utcStamp<<std::endl;
        os<<"	lat: "<<s.lat<<std::endl;
        os<<"	lon: "<<s.lon<<std::endl;
        os<<"	heigth: "<<s.heigth<<std::endl;
        os<<"	speed_hor: "<<s.speed_hor<<std::endl;
        os<<"	speed_ver: "<<s.speed_ver<<std::endl;
        os<<"	heading: "<<s.heading<<std::endl;
        os<<"	quality: "<<s.quality<<std::endl;
        os<<"	sats: "<<s.sats<<std::endl;
        os<<"	age: "<<s.age<<std::endl;
        os<<"	cov: "<<s.cov<<std::endl;
        os<<"	pressure: "<<s.pressure<<std::endl;
        os<<"	temp: "<<s.temp<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(14);
        structVars[0].set("header", header);
        structVars[1].set("utcStamp", utcStamp);
        structVars[2].set("lat", lat);
        structVars[3].set("lon", lon);
        structVars[4].set("heigth", heigth);
        structVars[5].set("speed_hor", speed_hor);
        structVars[6].set("speed_ver", speed_ver);
        structVars[7].set("heading", heading);
        structVars[8].set("quality", quality);
        structVars[9].set("sats", sats);
        structVars[10].set("age", age);
        structVars[11].set("cov", cov);
        structVars[12].set("pressure", pressure);
        structVars[13].set("temp", temp);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["utcStamp"].get(utcStamp);
        var["lat"].get(lat);
        var["lon"].get(lon);
        var["heigth"].get(heigth);
        var["speed_hor"].get(speed_hor);
        var["speed_ver"].get(speed_ver);
        var["heading"].get(heading);
        var["quality"].get(quality);
        var["sats"].get(sats);
        var["age"].get(age);
        var["cov"].get(cov);
        var["pressure"].get(pressure);
        var["temp"].get(temp);
        return true;
    }
};


}}
