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
    tk::math::Vec3<double> speed;
    tk::math::Vec3<double> angle;
    int quality;
    int sats;
    int age;
    tk::math::Mat3d cov;
    tk::math::Mat3d covSpeed;
    tk::math::Mat3d covAngle;
    double pressure;
    double temp;
    tk::common::Map<double> specific;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
        utcStamp = 0;
        lat = 0;
        lon = 0;
        heigth = 0;
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
        speed = s.speed;
        angle = s.angle;
        quality = s.quality;
        sats = s.sats;
        age = s.age;
        cov = s.cov;
        covSpeed = s.covSpeed;
        covAngle = s.covAngle;
        pressure = s.pressure;
        temp = s.temp;
        specific = s.specific;
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
        os<<"	speed: "<<s.speed<<std::endl;
        os<<"	angle: "<<s.angle<<std::endl;
        os<<"	quality: "<<s.quality<<std::endl;
        os<<"	sats: "<<s.sats<<std::endl;
        os<<"	age: "<<s.age<<std::endl;
        os<<"	cov: "<<s.cov<<std::endl;
        os<<"	covSpeed: "<<s.covSpeed<<std::endl;
        os<<"	covAngle: "<<s.covAngle<<std::endl;
        os<<"	pressure: "<<s.pressure<<std::endl;
        os<<"	temp: "<<s.temp<<std::endl;
        os<<"	specific: "<<s.specific<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(16);
        structVars[0].set("header", header);
        structVars[1].set("utcStamp", utcStamp);
        structVars[2].set("lat", lat);
        structVars[3].set("lon", lon);
        structVars[4].set("heigth", heigth);
        structVars[5].set("speed", speed);
        structVars[6].set("angle", angle);
        structVars[7].set("quality", quality);
        structVars[8].set("sats", sats);
        structVars[9].set("age", age);
        structVars[10].set("cov", cov);
        structVars[11].set("covSpeed", covSpeed);
        structVars[12].set("covAngle", covAngle);
        structVars[13].set("pressure", pressure);
        structVars[14].set("temp", temp);
        structVars[15].set("specific", specific);
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
        var["speed"].get(speed);
        var["angle"].get(angle);
        var["quality"].get(quality);
        var["sats"].get(sats);
        var["age"].get(age);
        var["cov"].get(cov);
        var["covSpeed"].get(covSpeed);
        var["covAngle"].get(covAngle);
        var["pressure"].get(pressure);
        var["temp"].get(temp);
        var["specific"].get(specific);
        return true;
    }
};


}}
