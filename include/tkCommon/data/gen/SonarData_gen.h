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
    
    void init() override
    {
        SensorData::init();
        header.type = type;
        resolution = 0;
        bits = 0;
        image.init();
    }
    SonarData_gen& operator=(const SonarData_gen& s)
    {
        SensorData::operator=(s);
        azimuth = s.azimuth;
        resolution = s.resolution;
        bits = s.bits;
        image = s.image;
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
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0].set("header", header);
        structVars[1].set("azimuth", azimuth);
        structVars[2].set("resolution", resolution);
        structVars[3].set("bits", bits);
        structVars[4].set("image", image);
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
        return true;
    }
};


}}
