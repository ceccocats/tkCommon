// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"


namespace tk { namespace data {

class KistlerData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec4<double> data;
    
    void init() override
    {
        SensorData::init();
        header.type = type;
    }
    KistlerData_gen& operator=(const KistlerData_gen& s)
    {
        SensorData::operator=(s);
        data = s.data;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, KistlerData_gen& s)
    {
        os<<"KistlerData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	data: "<<s.data<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(2);
        structVars[0].set("header", header);
        structVars[1].set("data", data);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["data"].get(data);
        return true;
    }
};


}}
