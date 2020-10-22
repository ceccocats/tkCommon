#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

/**
 * Actuation Data
 */

class ActuationData : public SensorData {
    public:
        double steerAngle;
        double accel;
        double speed;
        double torque;

        void init() {
            steerAngle = 0;
            accel = 0;
            speed = 0;
            torque = 0;
        }
        
        ActuationData& operator=(const ActuationData& a){
            SensorData::operator=(a);

            this->steerAngle = a.steerAngle;
            this->accel = a.accel;
            this->speed = a.speed;
            this->torque = a.torque;
            return *this;
        }
};

}}
