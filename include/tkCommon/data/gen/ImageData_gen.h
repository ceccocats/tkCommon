// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

template <class T>
class ImageData_gen : public SensorData
{
public:
    static const DataType type;
    T_to_class_type<T> T_type;
    tk::math::Vec<T> data;
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
    ImageData_gen<T>& operator=(const ImageData_gen<T>& s)
    {
        SensorData::operator=(s);
        data = s.data;
        width = s.width;
        height = s.height;
        channels = s.channels;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, ImageData_gen<T>& s)
    {
        os<<"ImageData_gen"<<std::endl;
        os<<"	type:  "; s.T_type.print(os); os<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	data: "<<s.data<<std::endl;
        os<<"	width: "<<s.width<<std::endl;
        os<<"	height: "<<s.height<<std::endl;
        os<<"	channels: "<<s.channels<<std::endl;
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
