// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class ImageData_gen : public SensorData
{
public:
    tk::math::Mat<uint8_t> data;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;
    
    void init() override
    {
        SensorData::init();
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
    friend std::ostream& operator<<(std::ostream& os, const ImageData_gen& s)
    {
        os<<"ImageData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
        os<<"data: "<<s.data<<std::endl;
        os<<"width: "<<s.width<<std::endl;
        os<<"height: "<<s.height<<std::endl;
        os<<"channels: "<<s.channels<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(5);
        structVars[0].set("header", header);
        structVars[1].set("data", data);
        structVars[2].set("width", width);
        structVars[3].set("height", height);
        structVars[4].set("channels", channels);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["data"].get(data);
        var["width"].get(width);
        var["height"].get(height);
        var["channels"].get(channels);
        return true;
    }
};

}}
