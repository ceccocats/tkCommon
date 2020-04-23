#pragma once
#include "CSpline2D.h"

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
    bool init(std::vector<tk::common::Vector3<float>> waypoints);

    tk::common::Vector3<float> calc_position(double s, double d=0);

    std::vector<tk::common::Vector3<float>> calc_points(double step);

private:
    /**
        calc s
    */
    VectorXd calc_s();
};

}}