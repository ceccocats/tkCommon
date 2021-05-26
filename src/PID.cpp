#include "tkCommon/PID.h"

namespace tk { namespace common {
    PID::PID() :
        _max(0),
        _min(0),
        _Kp(0),
        _Kd(0),
        _Ki(0),
        _pre_error(0),
        _integral(0)
    {
    }

    void 
    PID::init(double Kp, double Ki, double Kd, double min, double max) {
        _max        = max;
        _min        = min;
        _Kp         = Kp;
        _Kd         = Kd;
        _Ki         = Ki;
        _pre_error  = 0;
        _integral   = 0;
    }

    double 
    PID::calculate(double dt, double error) {
        // Proportional term
        double Pout = _Kp * error;

        // Integral term
        _integral += error * dt;
        double Iout = _Ki * _integral;

        // Derivative term
        double derivative = (error - _pre_error) / dt;
        double Dout = _Kd * derivative;

        // Calculate total output
        double output = Pout + Iout + Dout;

        // Restrict to max/min
        if( output > _max )
            output = _max;
        else if( output < _min )
            output = _min;

        // Save error to previous error
        _pre_error = error;

        //std::cout<<"PID Debug:\n\tdt: "<<dt<<"\n\tPout: "<<Pout<<"\n\tIout: "<<Iout<<"\n\tDout: "<<Dout<<"\n";

        return output;
    }
}}