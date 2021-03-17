// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"
#include "tkCommon/math/Mat.h"

namespace tk { namespace data {

class ImuData_gen : public SensorData
{
public:
    static const DataType type;
    tk::math::Vec3<double> acc;
    tk::math::Vec3<double> angleVel;
    tk::math::Vec3<double> angle;
    tk::math::Vec3<double> mag;
    tk::math::Mat3d covAcc;
    tk::math::Mat3d covAngleVel;
    tk::math::Mat3d covAngle;
    tk::math::Mat3d covMag;
    double sideSlip;
    
    void init() override
    {
        SensorData::init();
        sideSlip = 0;
    }
    ImuData_gen& operator=(const ImuData_gen& s)
    {
        SensorData::operator=(s);
        acc = s.acc;
        angleVel = s.angleVel;
        angle = s.angle;
        mag = s.mag;
        covAcc = s.covAcc;
        covAngleVel = s.covAngleVel;
        covAngle = s.covAngle;
        covMag = s.covMag;
        sideSlip = s.sideSlip;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, ImuData_gen& s)
    {
        os<<"ImuData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	header.fps:   "<<s.header.fps<<std::endl;
        os<<"	acc: "<<s.acc<<std::endl;
        os<<"	angleVel: "<<s.angleVel<<std::endl;
        os<<"	angle: "<<s.angle<<std::endl;
        os<<"	mag: "<<s.mag<<std::endl;
        os<<"	covAcc: "<<s.covAcc<<std::endl;
        os<<"	covAngleVel: "<<s.covAngleVel<<std::endl;
        os<<"	covAngle: "<<s.covAngle<<std::endl;
        os<<"	covMag: "<<s.covMag<<std::endl;
        os<<"	sideSlip: "<<s.sideSlip<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(10);
        structVars[0].set("header", header);
        structVars[1].set("acc", acc);
        structVars[2].set("angleVel", angleVel);
        structVars[3].set("angle", angle);
        structVars[4].set("mag", mag);
        structVars[5].set("covAcc", covAcc);
        structVars[6].set("covAngleVel", covAngleVel);
        structVars[7].set("covAngle", covAngle);
        structVars[8].set("covMag", covMag);
        structVars[9].set("sideSlip", sideSlip);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["acc"].get(acc);
        var["angleVel"].get(angleVel);
        var["angle"].get(angle);
        var["mag"].get(mag);
        var["covAcc"].get(covAcc);
        var["covAngleVel"].get(covAngleVel);
        var["covAngle"].get(covAngle);
        var["covMag"].get(covMag);
        var["sideSlip"].get(sideSlip);
        return true;
    }
};


}}
