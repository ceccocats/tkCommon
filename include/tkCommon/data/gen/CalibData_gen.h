// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Mat.h"


namespace tk { namespace data {

class CalibData_gen : public SensorData
{
public:
    tk::math::Mat<float> K;
    tk::math::Mat<float> D;
    tk::math::Mat<float> R;
    
    void init() override
    {
        SensorData::init();
        K.resize(3,3);
        D.resize(1,5);
        R.resize(3,3);
    }
    CalibData_gen& operator=(const CalibData_gen& s)
    {
        SensorData::operator=(s);
        K = s.K;
        D = s.D;
        R = s.R;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, CalibData_gen& s)
    {
        os<<"CalibData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	K: "<<s.K<<std::endl;
        os<<"	D: "<<s.D<<std::endl;
        os<<"	R: "<<s.R<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(4);
        structVars[0].set("header", header);
        structVars[1].set("K", K);
        structVars[2].set("D", D);
        structVars[3].set("R", R);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["K"].get(K);
        var["D"].get(D);
        var["R"].get(R);
        return true;
    }
};



}}
