// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

class DepthData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec<uint16_t> data;
    uint32_t width;
    uint32_t height;
    
    void init() override
    {
        SensorData::init();
        width = 0;
        height = 0;
    }
    DepthData_gen& operator=(const DepthData_gen& s)
    {
        SensorData::operator=(s);
        data = s.data;
        width = s.width;
        height = s.height;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, DepthData_gen& s)
    {
        os<<"DepthData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	data: "<<s.data<<std::endl;
        os<<"	width: "<<s.width<<std::endl;
        os<<"	height: "<<s.height<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("data", data);
        structVars[2].set("width", width);
        structVars[3].set("height", height);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["data"].get(data);
        var["width"].get(width);
        var["height"].get(height);
        return true;
    }
};


}}
