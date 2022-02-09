// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

class PoseData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec3<double> position;
    tk::math::Vec3<double> orientation;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    PoseData_gen& operator=(const PoseData_gen& s)
    {
        SensorData::operator=(s);
        position = s.position;
        orientation = s.orientation;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, PoseData_gen& s)
    {
        os<<"PoseData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	position: "<<s.position<<std::endl;
        os<<"	orientation: "<<s.orientation<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(3);
        structVars[0].set("header", header);
        structVars[1].set("position", position);
        structVars[2].set("orientation", orientation);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["position"].get(position);
        var["orientation"].get(orientation);
        return true;
    }
};


}}
