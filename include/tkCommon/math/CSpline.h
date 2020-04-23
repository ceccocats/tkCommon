#pragma once

#include <tkCommon/common.h>
using namespace Eigen;

namespace tk { namespace  math {

/**
    Cubic Spline class
*/
class CSpline {

public:

    // waypoints
    VectorXd x, y;

    int n;      // number of x waypoints
    VectorXd h; // diff beetween consecutive x
    VectorXd a, b, c, d;


    CSpline();

    virtual ~CSpline();

    /**
        init
        @param waypoints x, y
    */
    bool init(VectorXd _x, VectorXd _y);

    /**
        Calc position
        if t is outside of the input x, return None
    */
    double calc(double t);

    /**
        Calc first derivative
        if t is outside of the input x, return None
    */
    double calcd(double t);

    /**
        Calc second derivative
    */
    double calcdd(double t);

    /**
        search data segment index
    */
    int search_index(double x);
};

}}