// this file is generated DO NOT DIRECTLY MODIFY
#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/Vec.h"
#include "tkCommon/data/gen/OdomData_gen.h"

namespace tk { namespace data {

class VehicleData_gen : public SensorData
{
public:
    double CAR_WHEELBASE;
    tk::math::Vec3<double> CAR_DIM;
    double CAR_BACK2AXLE;
    double CAR_MASS;
    double CAR_FRONTAXLE_W;
    double CAR_BACKAXLE_W;
    double CAR_WHEEL_R;
    double speed;
    double speedKMH;
    double yawRate;
    double accX;
    double accY;
    double steerAngle;
    double steerAngleRate;
    double wheelAngle;
    int brakePedalStatus;
    double brakeMasterPressure;
    double gasPedal;
    double engineTorque;
    double engineFrictionTorque;
    int actualGear;
    int RPM;
    double wheelFLspeed;
    double wheelFRspeed;
    double wheelRLspeed;
    double wheelRRspeed;
    int wheelFLdir;
    int wheelFRdir;
    int wheelRLdir;
    int wheelRRdir;
    double sideSlip;
    int tractionGrip;
    tk::data::OdomData_gen odom;
    
    void init() override
    {
        SensorData::init();
        CAR_WHEELBASE = 0;
        CAR_BACK2AXLE = 0;
        CAR_MASS = 0;
        CAR_FRONTAXLE_W = 0;
        CAR_BACKAXLE_W = 0;
        CAR_WHEEL_R = 0;
        speed = 0;
        speedKMH = 0;
        yawRate = 0;
        accX = 0;
        accY = 0;
        steerAngle = 0;
        steerAngleRate = 0;
        wheelAngle = 0;
        brakePedalStatus = 0;
        brakeMasterPressure = 0;
        gasPedal = 0;
        engineTorque = 0;
        engineFrictionTorque = 0;
        actualGear = 0;
        RPM = 0;
        wheelFLspeed = 0;
        wheelFRspeed = 0;
        wheelRLspeed = 0;
        wheelRRspeed = 0;
        wheelFLdir = 0;
        wheelFRdir = 0;
        wheelRLdir = 0;
        wheelRRdir = 0;
        sideSlip = 0;
        tractionGrip = 0;
        odom.init();;
    }
    VehicleData_gen& operator=(const VehicleData_gen& s)
    {
        SensorData::operator=(s);
        CAR_WHEELBASE = s.CAR_WHEELBASE;
        CAR_DIM = s.CAR_DIM;
        CAR_BACK2AXLE = s.CAR_BACK2AXLE;
        CAR_MASS = s.CAR_MASS;
        CAR_FRONTAXLE_W = s.CAR_FRONTAXLE_W;
        CAR_BACKAXLE_W = s.CAR_BACKAXLE_W;
        CAR_WHEEL_R = s.CAR_WHEEL_R;
        speed = s.speed;
        speedKMH = s.speedKMH;
        yawRate = s.yawRate;
        accX = s.accX;
        accY = s.accY;
        steerAngle = s.steerAngle;
        steerAngleRate = s.steerAngleRate;
        wheelAngle = s.wheelAngle;
        brakePedalStatus = s.brakePedalStatus;
        brakeMasterPressure = s.brakeMasterPressure;
        gasPedal = s.gasPedal;
        engineTorque = s.engineTorque;
        engineFrictionTorque = s.engineFrictionTorque;
        actualGear = s.actualGear;
        RPM = s.RPM;
        wheelFLspeed = s.wheelFLspeed;
        wheelFRspeed = s.wheelFRspeed;
        wheelRLspeed = s.wheelRLspeed;
        wheelRRspeed = s.wheelRRspeed;
        wheelFLdir = s.wheelFLdir;
        wheelFRdir = s.wheelFRdir;
        wheelRLdir = s.wheelRLdir;
        wheelRRdir = s.wheelRRdir;
        sideSlip = s.sideSlip;
        tractionGrip = s.tractionGrip;
        odom = s.odom;
        return *this;
    }
    friend std::ostream& operator<<(std::ostream& os, VehicleData_gen& s)
    {
        os<<"VehicleData_gen"<<std::endl;
        os<<"	header.name:  "<<s.header.name<<std::endl;
        os<<"	header.stamp: "<<s.header.stamp<<std::endl;
        os<<"	CAR_WHEELBASE: "<<s.CAR_WHEELBASE<<std::endl;
        os<<"	CAR_DIM: "<<s.CAR_DIM<<std::endl;
        os<<"	CAR_BACK2AXLE: "<<s.CAR_BACK2AXLE<<std::endl;
        os<<"	CAR_MASS: "<<s.CAR_MASS<<std::endl;
        os<<"	CAR_FRONTAXLE_W: "<<s.CAR_FRONTAXLE_W<<std::endl;
        os<<"	CAR_BACKAXLE_W: "<<s.CAR_BACKAXLE_W<<std::endl;
        os<<"	CAR_WHEEL_R: "<<s.CAR_WHEEL_R<<std::endl;
        os<<"	speed: "<<s.speed<<std::endl;
        os<<"	speedKMH: "<<s.speedKMH<<std::endl;
        os<<"	yawRate: "<<s.yawRate<<std::endl;
        os<<"	accX: "<<s.accX<<std::endl;
        os<<"	accY: "<<s.accY<<std::endl;
        os<<"	steerAngle: "<<s.steerAngle<<std::endl;
        os<<"	steerAngleRate: "<<s.steerAngleRate<<std::endl;
        os<<"	wheelAngle: "<<s.wheelAngle<<std::endl;
        os<<"	brakePedalStatus: "<<s.brakePedalStatus<<std::endl;
        os<<"	brakeMasterPressure: "<<s.brakeMasterPressure<<std::endl;
        os<<"	gasPedal: "<<s.gasPedal<<std::endl;
        os<<"	engineTorque: "<<s.engineTorque<<std::endl;
        os<<"	engineFrictionTorque: "<<s.engineFrictionTorque<<std::endl;
        os<<"	actualGear: "<<s.actualGear<<std::endl;
        os<<"	RPM: "<<s.RPM<<std::endl;
        os<<"	wheelFLspeed: "<<s.wheelFLspeed<<std::endl;
        os<<"	wheelFRspeed: "<<s.wheelFRspeed<<std::endl;
        os<<"	wheelRLspeed: "<<s.wheelRLspeed<<std::endl;
        os<<"	wheelRRspeed: "<<s.wheelRRspeed<<std::endl;
        os<<"	wheelFLdir: "<<s.wheelFLdir<<std::endl;
        os<<"	wheelFRdir: "<<s.wheelFRdir<<std::endl;
        os<<"	wheelRLdir: "<<s.wheelRLdir<<std::endl;
        os<<"	wheelRRdir: "<<s.wheelRRdir<<std::endl;
        os<<"	sideSlip: "<<s.sideSlip<<std::endl;
        os<<"	tractionGrip: "<<s.tractionGrip<<std::endl;
        os<<"	odom: "<<s.odom<<std::endl;
        return os;
    }
    bool toVar(std::string name, tk::math::MatIO::var_t &var)
    {
        std::vector<tk::math::MatIO::var_t> structVars(34);
        structVars[0].set("header", header);
        structVars[1].set("CAR_WHEELBASE", CAR_WHEELBASE);
        structVars[2].set("CAR_DIM", CAR_DIM);
        structVars[3].set("CAR_BACK2AXLE", CAR_BACK2AXLE);
        structVars[4].set("CAR_MASS", CAR_MASS);
        structVars[5].set("CAR_FRONTAXLE_W", CAR_FRONTAXLE_W);
        structVars[6].set("CAR_BACKAXLE_W", CAR_BACKAXLE_W);
        structVars[7].set("CAR_WHEEL_R", CAR_WHEEL_R);
        structVars[8].set("speed", speed);
        structVars[9].set("speedKMH", speedKMH);
        structVars[10].set("yawRate", yawRate);
        structVars[11].set("accX", accX);
        structVars[12].set("accY", accY);
        structVars[13].set("steerAngle", steerAngle);
        structVars[14].set("steerAngleRate", steerAngleRate);
        structVars[15].set("wheelAngle", wheelAngle);
        structVars[16].set("brakePedalStatus", brakePedalStatus);
        structVars[17].set("brakeMasterPressure", brakeMasterPressure);
        structVars[18].set("gasPedal", gasPedal);
        structVars[19].set("engineTorque", engineTorque);
        structVars[20].set("engineFrictionTorque", engineFrictionTorque);
        structVars[21].set("actualGear", actualGear);
        structVars[22].set("RPM", RPM);
        structVars[23].set("wheelFLspeed", wheelFLspeed);
        structVars[24].set("wheelFRspeed", wheelFRspeed);
        structVars[25].set("wheelRLspeed", wheelRLspeed);
        structVars[26].set("wheelRRspeed", wheelRRspeed);
        structVars[27].set("wheelFLdir", wheelFLdir);
        structVars[28].set("wheelFRdir", wheelFRdir);
        structVars[29].set("wheelRLdir", wheelRLdir);
        structVars[30].set("wheelRRdir", wheelRRdir);
        structVars[31].set("sideSlip", sideSlip);
        structVars[32].set("tractionGrip", tractionGrip);
        structVars[33].set("odom", odom);
        return var.setStruct(name, structVars);
    }
    bool fromVar(tk::math::MatIO::var_t &var)
    {
        if(var.empty()) return false;
        var["header"].get(header);
        var["CAR_WHEELBASE"].get(CAR_WHEELBASE);
        var["CAR_DIM"].get(CAR_DIM);
        var["CAR_BACK2AXLE"].get(CAR_BACK2AXLE);
        var["CAR_MASS"].get(CAR_MASS);
        var["CAR_FRONTAXLE_W"].get(CAR_FRONTAXLE_W);
        var["CAR_BACKAXLE_W"].get(CAR_BACKAXLE_W);
        var["CAR_WHEEL_R"].get(CAR_WHEEL_R);
        var["speed"].get(speed);
        var["speedKMH"].get(speedKMH);
        var["yawRate"].get(yawRate);
        var["accX"].get(accX);
        var["accY"].get(accY);
        var["steerAngle"].get(steerAngle);
        var["steerAngleRate"].get(steerAngleRate);
        var["wheelAngle"].get(wheelAngle);
        var["brakePedalStatus"].get(brakePedalStatus);
        var["brakeMasterPressure"].get(brakeMasterPressure);
        var["gasPedal"].get(gasPedal);
        var["engineTorque"].get(engineTorque);
        var["engineFrictionTorque"].get(engineFrictionTorque);
        var["actualGear"].get(actualGear);
        var["RPM"].get(RPM);
        var["wheelFLspeed"].get(wheelFLspeed);
        var["wheelFRspeed"].get(wheelFRspeed);
        var["wheelRLspeed"].get(wheelRLspeed);
        var["wheelRRspeed"].get(wheelRRspeed);
        var["wheelFLdir"].get(wheelFLdir);
        var["wheelFRdir"].get(wheelFRdir);
        var["wheelRLdir"].get(wheelRLdir);
        var["wheelRRdir"].get(wheelRRdir);
        var["sideSlip"].get(sideSlip);
        var["tractionGrip"].get(tractionGrip);
        var["odom"].get(odom);
        return true;
    }
};


}}
