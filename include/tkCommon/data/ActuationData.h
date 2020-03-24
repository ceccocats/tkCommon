#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

/**
 * Actuation Data
 */

class ActuationData{
    public:
        double steerAngle;
        double accelX;
        double speed;
        double torque;

        void init(double steerAngle, double accelX, double speed, double torque)
	    : steerAngle(steerAngle)
        , accelX(accelX)
        , speed(speed)
        , torque(torque){};

        ActuationData& operator=(const ActuationData& a){
            this->steerAngle = a.steerAngle;
            this->accelX = a.accelX;
            this->speed = a.speed;
            this->torque = a.torque;

            return *this;
        }
};

}}
