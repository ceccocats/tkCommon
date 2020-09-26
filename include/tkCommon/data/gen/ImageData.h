// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

class ImageData : public SensorData
{
public:
    uint8_t data;
    
    void init() override
    {
        SensorData::init();
        data = 0;
    }
    ImageData& operator=(const ImageData& s)
    {
        SensorData::operator=(s);
        data = s.data;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const ImageData& s)
    {
        os<<"ImageData:"<<std::endl;
        os<<"data: "<<s.data<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        tk::math::MatIO::var_t hvar;
        tk::data::SensorData::toVar("header", hvar);
        std::vector<tk::math::MatIO::var_t> structVars(2);
        structVars[0] = hvar;
        structVars[1].set("data", data);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        tk::data::SensorData::fromVar(var["header"]);
        var["data"].get(data);
        return true;
    }
};

}}
