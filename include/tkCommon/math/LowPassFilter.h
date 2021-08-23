#pragma once

#include <tkCommon/common.h>

namespace tk { namespace  math {

/**
    Second order low pass filter
*/
class LowPassFilter {

public:
    float a1 = 1.0;
    float a2 = 0;
    float a3 = 0;
    float b1 = 0;
    float b2 = 0;
    float b3 = 0;

    float u[3]; // 0 = n, 1 = (n-1), 2 = (n-2)
    float y[3]; // 0 = n, 1 = (n-1), 2 = (n-2)
    
    LowPassFilter() {}

    ~LowPassFilter() {}

    void init(float a1, float a2, float a3, float b1, float b2, float b3) {
        this->a1 = a1;
        this->a2 = a2;
        this->a3 = a3;
        this->b1 = b1;
        this->b2 = b2;
        this->b3 = b3;
        u[0] = u[1] = u[2] = y[0] = y[1] = y[2] = 0; 
    }

    float forward(float val) {
        u[2] = u[1];
        y[2] = y[1];
        u[1] = u[0];
        y[1] = y[0];
        u[0] = val;
        y[0] = (-a2*y[1] -a3*y[2] + b1*u[0] + b2*u[1] + b3*u[2])/a1;
    }
};

}}