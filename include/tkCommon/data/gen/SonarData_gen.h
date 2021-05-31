// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/gen/ImageData_gen.h"

namespace tk { namespace data {

class SonarData_gen : public SensorData
{
public:
    static const DataType type;
    tk::data::ImageData_gen<float> image;
    tk::math::Vec<float> azimuth;
    float resolution;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
        resolution = 0;
    }
    SonarData_gen& operator=(const SonarData_gen& s)
    {
        SensorData::operator=(s);
        image = s.image;
        azimuth = s.azimuth;
        resolution = s.resolution;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, SonarData_gen& s)
    {
        os<<"SonarData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	image: "<<s.image<<std::endl;
        os<<"	azimuth: "<<s.azimuth<<std::endl;
        os<<"	resolution: "<<s.resolution<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("image", image);
        structVars[2].set("azimuth", azimuth);
        structVars[3].set("resolution", resolution);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["image"].get(image);
        var["azimuth"].get(azimuth);
        var["resolution"].get(resolution);
        return true;
    }
};


}}
