#include "tkCommon/math/CSpline2D.h"
#include <iostream>

namespace tk { namespace  math {

CSpline2D::CSpline2D() {}

CSpline2D::~CSpline2D() {}

bool CSpline2D::init(std::vector<tk::common::Vector2<float>> waypoints) {

    // number of x waypoints
    nx = waypoints.size();

    x = VectorXd(nx);
    y = VectorXd(nx);
    for (int i = 0; i < nx; i++) {
        x(i) = waypoints[i].x;
        y(i) = waypoints[i].y;
    }

    s = calc_s();
    sx.init(s, x);
    sy.init(s, y);

    s_start = s(0);
    s_end = s(s.size() - 1);
    return true;
}

VectorXd CSpline2D::calc_s() {

    // build diff vector
    VectorXd dx = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        dx(i) = x(i + 1) - x(i);

    // build diff vector
    VectorXd dy = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        dy(i) = y(i + 1) - y(i);

    VectorXd ds = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        ds(i) = sqrt(dx(i) * dx(i) + dy(i) * dy(i));

    // comulative sum
    VectorXd s = VectorXd(nx);
    s(0) = 0.0;
    for (int i = 0; i < nx - 1; i++)
        s(i + 1) = s(i) + ds(i);

    return s;
}


tk::common::Vector2<float> CSpline2D::calc_position(double s, double d) {

    tk::common::Vector2<float> p;

    if(d!=0) {
        double yaw = calc_yaw(s);
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double x = 0;
        double y = -d;
        p.x = x*cos_yaw - y*sin_yaw;
        p.y = x*sin_yaw + y*cos_yaw;
    }

    p.x += sx.calc(s);
    p.y += sy.calc(s);
    return p;
}

double CSpline2D::calc_curvature(double s) {
    double dx = sx.calcd(s);
    double ddx = sx.calcdd(s);
    double dy = sy.calcd(s);
    double ddy = sy.calcdd(s);
    double k = (ddy * dx - ddx * dy) / (dx * dx + dy * dy);
    return k;
}

double CSpline2D::calc_yaw(double s) {
    double dx = sx.calcd(s);
    double dy = sy.calcd(s);
    double yaw = atan2(dy, dx);
    return yaw;
}

}}