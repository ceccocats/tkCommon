// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/data/PoseData.h"


namespace tk { namespace data {

class PathData_gen : public SensorData
{
public:
    static const DataType type;
    std::vector<tk::data::PoseData> poses;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    PathData_gen& operator=(const PathData_gen& s)
    {
        SensorData::operator=(s);
        poses = s.poses;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, PathData_gen& s)
    {
        os<<"PathData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(1);
        structVars[0].set("header", header);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        return true;
    }
};


}}
