// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/ImageData.h"

namespace tk { namespace data {

class SonarData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec<float> azimuth;
    float resolution;
    int bits;
    tk::data::ImageDataF image;
    tk::data::ImageDataF raw;
    double roll;
    double pitch;
    double yaw;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
        resolution = 0;
        bits = 0;
        image.init();
        image.init();
        roll = 0;
        pitch = 0;
        yaw = 0;
    }
    SonarData_gen& operator=(const SonarData_gen& s)
    {
        SensorData::operator=(s);
        azimuth = s.azimuth;
        resolution = s.resolution;
        bits = s.bits;
        image = s.image;
        raw = s.raw;
        roll = s.roll;
        pitch = s.pitch;
        yaw = s.yaw;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, SonarData_gen& s)
    {
        os<<"SonarData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	azimuth: "<<s.azimuth<<std::endl;
        os<<"	resolution: "<<s.resolution<<std::endl;
        os<<"	bits: "<<s.bits<<std::endl;
        os<<"	image: "<<s.image<<std::endl;
        os<<"	raw: "<<s.raw<<std::endl;
        os<<"	roll: "<<s.roll<<std::endl;
        os<<"	pitch: "<<s.pitch<<std::endl;
        os<<"	yaw: "<<s.yaw<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(9);
        structVars[0].set("header", header);
        structVars[1].set("azimuth", azimuth);
        structVars[2].set("resolution", resolution);
        structVars[3].set("bits", bits);
        structVars[4].set("image", image);
        structVars[5].set("raw", raw);
        structVars[6].set("roll", roll);
        structVars[7].set("pitch", pitch);
        structVars[8].set("yaw", yaw);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["azimuth"].get(azimuth);
        var["resolution"].get(resolution);
        var["bits"].get(bits);
        var["image"].get(image);
        var["raw"].get(raw);
        var["roll"].get(roll);
        var["pitch"].get(pitch);
        var["yaw"].get(yaw);
        return true;
    }
};


}}
