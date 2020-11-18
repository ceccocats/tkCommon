// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class ImageData_gen : public SensorData
{
public:
    uint8_t* data;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    
    void init() override
    {
        SensorData::init();
        width = 0;
        height = 0;
        channels = 0;
    }
    ImageData_gen& operator=(const ImageData_gen& s)
    {
        SensorData::operator=(s);
        data = s.data;
        width = s.width;
        height = s.height;
        channels = s.channels;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, ImageData_gen& s)
    {
        os<<"ImageData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	width: "<<s.width<<std::endl;
        os<<"	height: "<<s.height<<std::endl;
        os<<"	channels: "<<s.channels<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("width", width);
        structVars[2].set("height", height);
        structVars[3].set("channels", channels);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["width"].get(width);
        var["height"].get(height);
        var["channels"].get(channels);
        return true;
    }
};


}}
