#include "tkCommon/math/CSpline.h"
#include <iostream>

namespace tk { namespace  math {

    CSpline::CSpline() {}

    CSpline::~CSpline() {}

    bool CSpline::init(VectorXd _x, VectorXd _y) {
        x = _x;
        y = _y;

        // number of x waypoints
        nx = x.size();

        // build diff vector
        h = VectorXd(nx - 1);
        for (int i = 0; i < nx - 1; i++)
            h(i) = x(i + 1) - x(i);

        // calc coefficient c
        a = y;
        A = calc_A(h);
        B = calc_B(h);
        c = A.colPivHouseholderQr().solve(B);

        // calc spline coefficient b and d
        d = VectorXd(nx - 1);
        b = VectorXd(nx - 1);
        for (int i = 0; i < nx - 1; i++) {
            d(i) = (c(i + 1) - c(i)) / (3.0 * h(i));
            double tb = (a(i + 1) - a(i)) / h(i) - h(i) * (c(i + 1) + 2.0 * c(i)) / 3.0;
            b(i) = tb;
        }

        return true;
    }

    double CSpline::calc(double t) {

        int i = search_index(t);
        double dx = t - x(i);

        return a(i) + b(i) * dx + c(i) * dx * dx + d(i) * dx * dx * dx;
    }

    double CSpline::calcd(double t) {

        int i = search_index(t);
        double dx = t - x(i);

        return b(i) + 2.0 * c(i) * dx + 3.0 * d(i) * dx * dx;
    }

    double CSpline::calcdd(double t) {

        int i = search_index(t);
        double dx = t - x(i);

        return 2.0 * c(i) + 6.0 * d(i) * dx;
    }

    int CSpline::search_index(double sx) {

        for (int i = 1; i < nx - 1; i++) {
            if (sx < x(i))
                return i - 1;
        }
        if (nx > 1)
            return nx - 2;
        return 0;
    }

    MatrixXd CSpline::calc_A(VectorXd h) {

        A = MatrixXd::Zero(nx, nx);
        A(0, 0) = 1.0;
        for (int i = 0; i < nx - 1; i++) {
            if (i != nx - 2)
                A(i + 1, i + 1) = 2.0 * (h(i) + h(i + 1));
            A(i + 1, i) = h(i);
            A(i, i + 1) = h(i);
        }

        A(0, 1) = 0.0;
        A(nx - 1, nx - 2) = 0.0;
        A(nx - 1, nx - 1) = 1.0;
        return A;
    }

    VectorXd CSpline::calc_B(VectorXd h) {

        B = VectorXd::Zero(nx);
        for (int i = 0; i < nx - 2; i++) {
            B(i + 1) = 3.0 * (a(i + 2) - a(i + 1)) / h(i + 1) - 3.0 * (a(i + 1) - a(i)) / h(i);
        }
        return B;
    }

}}