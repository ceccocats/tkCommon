#pragma once 

#include "tkCommon/common.h"

namespace tk { namespace common {
    class PID {
    private:
        double _max;
        double _min;
        double _Kp;
        double _Kd;
        double _Ki;
        double _pre_error;
        double _integral;

    public:
        PID();
        ~PID() = default;
        
        void init(double Kp, double Ki, double Kd, double min, double max);

        void setGain(double Kp, double Ki, double Kd) { _Kp = Kp; _Ki = Ki; _Kd = Kd; }
        void resetIntegral() { _integral = 0; }

        double calculate(double dt, double error);
    };    
}}