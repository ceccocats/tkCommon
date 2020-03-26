#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

/**
 * Actuation Data
 */

class ActuationData{
    public:
        double steerAngle;
        double accel;
        double speed;
        double torque;

        void init(double steerAngle, double accel, double speed, double torque)
	    : steerAngle(steerAngle)
        , accel(accel)
        , speed(speed)
        , torque(torque){};

        ActuationData& operator=(const ActuationData& a){
            this->steerAngle = a.steerAngle;
            this->accel = a.accel;
            this->speed = a.speed;
            this->torque = a.torque;

            return *this;
        }
};

}}
