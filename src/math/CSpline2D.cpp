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
        x(i) = waypoints[i].x();
        y(i) = waypoints[i].y();
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
        p.x() = x*cos_yaw - y*sin_yaw;
        p.y() = x*sin_yaw + y*cos_yaw;
    }

    p.x() += sx.calc(s);
    p.y() += sy.calc(s);
    return p;
}

tk::common::Vector2<float> 
CSpline2D::calc_projection(tk::common::Vector2<float> pose, double s_guess)
{
    tk::common::Vector2<float> tmp = calc_position(s_guess);
    double dist_pow = pow(tmp.x() - pose.x(), 2) + pow(tmp.y() - pose.y(), 2);

	if (dist_pow > pow(5.0, 2)) {
		double min_dist = std::numeric_limits<double>::max();
		double min_index = 0;
		for (size_t i = 0; i < this->x.size(); ++i) {
			dist_pow = pow(this->x[i] - pose.x(), 2) + pow(this->y[i] - pose.y(), 2);
			if (dist_pow < min_dist) {
				min_dist = dist_pow;
				min_index = i;
			}
		}
		tmp.x() = this->x[min_index];
		tmp.y() = this->y[min_index];
		s_guess = this->s[min_index];
	}

	double s_opt = s_guess;
	double s_previous = s_opt;

	for(int i=0; i<30; ++i)
	{
		tmp = calc_position(s_opt);
		double dx = sx.calcd(s_opt);
		double ddx = sx.calcdd(s_opt);

		double dy = sy.calcd(s_opt);
		double ddy = sy.calcdd(s_opt);

		double diff_x = tmp.x() - pose.x();
		double diff_y = tmp.y() - pose.y();

		double jac = 2.0 * diff_x * dx + 2.0 * diff_y * dy;
        double hessian = 2.0 * dx * dx + 2.0 * diff_x * ddx +
                         2.0 * dy * dy + 2.0 * diff_y * ddy;

        // giove temp fix
        if (hessian < 1e-20f) 
			break;

        // Newton method
        s_opt -= jac/hessian;
        s_opt = s_opt - this->s_end*std::floor(s_opt/this->s_end);

        if(std::abs(s_previous - s_opt) <= 1e-5)
			break;
		
        s_previous = s_opt;
	}

    float d = calc_position(s_opt).dist(tk::common::Vector2<float>{pose.x(), pose.y()}, 2);

    return tk::common::Vector2<float>{(float) s_opt, (float) d};
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