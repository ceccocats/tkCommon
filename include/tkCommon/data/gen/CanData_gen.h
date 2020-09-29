// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include <linux/can.h>
#include <linux/can/raw.h>

namespace tk { namespace data {

class CanData_gen : public SensorData
{
public:
    struct can_frame frame;
    
    void init() override
    {
        SensorData::init();
    }
    CanData_gen& operator=(const CanData_gen& s)
    {
        SensorData::operator=(s);
        frame = s.frame;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, const CanData_gen& s)
    {
        os<<"CanData_gen:"<<std::endl;
        os<<"header.stamp:"<<s.header.stamp<<std::endl;
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
