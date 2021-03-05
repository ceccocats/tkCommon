// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class CalibData_gen : public SensorData
{
public:
    static const DataType type;
    int w;
    int h;
    tk::math::Mat<float> k;
    tk::math::Mat<float> d;
    tk::math::Mat<float> r;
    
    void init() override
    {
        SensorData::init();
        w = 0;
        h = 0;
        k.resize(3,3);
        d.resize(1,5);
        r.resize(3,3);
    }
    CalibData_gen& operator=(const CalibData_gen& s)
    {
        SensorData::operator=(s);
        w = s.w;
        h = s.h;
        k = s.k;
        d = s.d;
        r = s.r;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, CalibData_gen& s)
    {
        os<<"CalibData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	w: "<<s.w<<std::endl;
        os<<"	h: "<<s.h<<std::endl;
        os<<"	k: "<<s.k<<std::endl;
        os<<"	d: "<<s.d<<std::endl;
        os<<"	r: "<<s.r<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(6);
        structVars[0].set("header", header);
        structVars[1].set("w", w);
        structVars[2].set("h", h);
        structVars[3].set("k", k);
        structVars[4].set("d", d);
        structVars[5].set("r", r);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["w"].get(w);
        var["h"].get(h);
        var["k"].get(k);
        var["d"].get(d);
        var["r"].get(r);
        return true;
    }
};


}}
