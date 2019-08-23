#pragma once
#undef Success  // stupid X11
#include "Eigen/Dense"
using namespace Eigen;
#include "tkCommon/common.h"

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

private:
    /**
        calc matrix A for spline coefficient c
    */
    MatrixXd calc_A(VectorXd h);

    /**
        calc matrix B for spline coefficient c
    */
    VectorXd calc_B(VectorXd h);
};

}}