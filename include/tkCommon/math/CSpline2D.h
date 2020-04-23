#pragma once
#include "CSpline.h"

using namespace Eigen;

namespace tk { namespace  math {

/**
    Cubic Spline2D class
*/
class CSpline2D {

public:

    // waypoints
    VectorXd x, y;
    int nx;

    VectorXd s;
    CSpline sx, sy;
    double s_start, s_end;


    CSpline2D();

    virtual ~CSpline2D();

    /**
        init
        @param waypoints x, y
    */
    bool init(std::vector<tk::common::Vector2<float>> waypoints);

    tk::common::Vector2<float> calc_position(double s, double d=0);

    double calc_curvature(double s);

    double calc_yaw(double s);

private:
    /**
        calc s
    */
    VectorXd calc_s();
};

}}