#pragma once
#include "CSpline2D.h"

#undef Success  // stupid X11
#include "Eigen/Dense"
using namespace Eigen;

namespace tk { namespace  math {

/**
    Cubic Spline3D class
*/
class CSpline3D : public CSpline2D {

public:

    // waypoints
    VectorXd z;

    CSpline sz;

    CSpline3D();

    virtual ~CSpline3D();

    /**
        init
        @param waypoints x, y, z
    */
    bool init(std::vector<Vector3f> waypoints);

    Vector3f calc_position(double s);

private:
    /**
        calc s
    */
    VectorXd calc_s();
};

}}