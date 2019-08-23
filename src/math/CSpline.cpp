#include "tkCommon/math/CSpline.h"
#include <iostream>

namespace tk { namespace  math {

    CSpline::CSpline() {}

    CSpline::~CSpline() {}

    bool CSpline::init(VectorXd _x, VectorXd _y) {
        x = _x;
        y = _y;

        // number of x waypoints
        n = x.size();

        // init coeffs
        h = VectorXd(n - 1);
        a = VectorXd(n - 1);
        b = VectorXd(n - 1);
        c = VectorXd(n - 1);
        d = VectorXd(n - 1);

        VectorXd F = VectorXd(n);
        MatrixXd m = MatrixXd::Zero(n, n);
        VectorXd s = VectorXd::Zero(n);

        for(int i=n-1; i>0; i--){
            h(i-1) = x(i)-x(i-1);

            // avoid NAN
            if(h(i-1) == 0)
                F(i) = 0;
            else
                F(i)  = (y(i)-y(i-1))/(h(i-1));
        }

        // formation of h, s , f matrix
        for(int i=1; i<n-1; i++){
            m(i,i) = 2*(h(i-1)+h(i));
            if(i!=1){
                m(i,  i-1) = h(i-1);
                m(i-1,i  ) = h(i-1);
            }
            m(i,n-1) = 6*(F(i+1)-F(i));
        }

        // forward elimination
        for(int i=1; i<n-2; i++){
            double temp = (m(i+1, i)/m(i,i));
            for(int j=1; j<=n-1; j++)
                m(i+1,j) -= temp*m(i,j);
        }

        // backward substitution
        for(int i=n-2; i>0; i--){
            double accum =0;
            for(int j=i; j<=n-2; j++) {
                accum += m(i, j) * s(j);
            }
            s(i)=(m(i,n-1)-accum)/m(i,i);
        }

        // calc spline coefficient b and d
        for (int i = 0; i < n - 1; i++) {
            d(i) = (s(i+1)-s(i))/(6*h(i));
            c(i) = s(i)/2;
            b(i) = (y(i+1)-y(i))/h(i)-(2*h(i)*s(i)+s(i+1)*h(i))/6;
            a(i) = y(i);
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

        for (int i = 1; i < n - 1; i++) {
            if (sx < x(i))
                return i - 1;
        }
        if (n > 1)
            return n - 2;
        return 0;
    }


}}