// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/DepthData.h"

#include "tkCommon/data/ImageData.h"


namespace tk { namespace data {

class StereoData_gen : public SensorData
{
public:
    static const DataType type;
    int width;
    int height;
    int channels;
    int c_width;
    int c_height;
    int c_channels;
    int d_width;
    int d_height;
    
    void init() override
    {
        SensorData::init();
        width = 0;
        height = 0;
        channels = 0;
        c_width = 0;
        c_height = 0;
        c_channels = 0;
        d_width = 0;
        d_height = 0;
    }
    StereoData_gen& operator=(const StereoData_gen& s)
    {
        SensorData::operator=(s);
        width = s.width;
        height = s.height;
        channels = s.channels;
        c_width = s.c_width;
        c_height = s.c_height;
        c_channels = s.c_channels;
        d_width = s.d_width;
        d_height = s.d_height;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, StereoData_gen& s)
    {
        os<<"StereoData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	width: "<<s.width<<std::endl;
        os<<"	height: "<<s.height<<std::endl;
        os<<"	channels: "<<s.channels<<std::endl;
        os<<"	c_width: "<<s.c_width<<std::endl;
        os<<"	c_height: "<<s.c_height<<std::endl;
        os<<"	c_channels: "<<s.c_channels<<std::endl;
        os<<"	d_width: "<<s.d_width<<std::endl;
        os<<"	d_height: "<<s.d_height<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(9);
        structVars[0].set("header", header);
        structVars[1].set("width", width);
        structVars[2].set("height", height);
        structVars[3].set("channels", channels);
        structVars[4].set("c_width", c_width);
        structVars[5].set("c_height", c_height);
        structVars[6].set("c_channels", c_channels);
        structVars[7].set("d_width", d_width);
        structVars[8].set("d_height", d_height);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["width"].get(width);
        var["height"].get(height);
        var["channels"].get(channels);
        var["c_width"].get(c_width);
        var["c_height"].get(c_height);
        var["c_channels"].get(c_channels);
        var["d_width"].get(d_width);
        var["d_height"].get(d_height);
        return true;
    }
};


}}
