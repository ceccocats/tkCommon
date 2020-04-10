#include "tkCommon/math/CSpline3D.h"
#include <iostream>

namespace tk { namespace  math {

CSpline3D::CSpline3D() {}

CSpline3D::~CSpline3D() {}

bool CSpline3D::init(std::vector<tk::common::Vector3<float>> waypoints) {

    // number of x waypoints
    nx = waypoints.size();

    x = VectorXd(nx);
    y = VectorXd(nx);
    z = VectorXd(nx);
    for (int i = 0; i < nx; i++) {
        x(i) = waypoints[i].x;
        y(i) = waypoints[i].y;
        z(i) = waypoints[i].z;
    }

    s = calc_s();
    sx.init(s, x);
    sy.init(s, y);
    sz.init(s, z);

    s_start = s(0);
    s_end = s(s.size() - 1);
    return true;
}

VectorXd CSpline3D::calc_s() {

    // build diff vector
    VectorXd dx = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        dx(i) = x(i + 1) - x(i);

    // build diff vector
    VectorXd dy = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        dy(i) = y(i + 1) - y(i);

    // build diff vector
    VectorXd dz = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        dz(i) = z(i + 1) - z(i);


    VectorXd ds = VectorXd(nx - 1);
    for (int i = 0; i < nx - 1; i++)
        ds(i) = sqrt(dx(i) * dx(i) + dy(i) * dy(i) + dz(i) * dz(i));

    // comulative sum
    VectorXd s = VectorXd(nx);
    s(0) = 0.0;
    for (int i = 0; i < nx - 1; i++)
        s(i + 1) = s(i) + ds(i);

    return s;
}

tk::common::Vector3<float> CSpline3D::calc_position(double s, double d) {

    tk::common::Vector3<float> p;

    if(d!=0) {
        double yaw = calc_yaw(s);
        double sin_yaw = sin(yaw);
        double cos_yaw = cos(yaw);
        double x = 0;
        double y = -d;
        p.x = x*cos_yaw - y*sin_yaw;
        p.y = x*sin_yaw + y*cos_yaw;
        p.z = 0;
    }

    p.x += sx.calc(s);
    p.y += sy.calc(s);
    p.z += sz.calc(s);
    return p;
}

std::vector<tk::common::Vector3<float>> CSpline3D::calc_points(double step) {

    std::vector<tk::common::Vector3<float>> pts;

    for(float s = s_start; s<s_end; s+=step) {
        pts.push_back(calc_position(s));
    }
    return pts;
}


}}