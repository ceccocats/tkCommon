#pragma once

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace tk {
namespace math {
/**
 * @brief univariate gaussian PDF
 */
template<typename T>
T gaussianProbUni(T mean, T var, T x){
    const T dif = x - mean;
    return 1.0 / sqrt(2.0 * M_PI * var) * exp(-(dif * dif) / (2 * var));
}

/**
 * @brief multivariate gaussian PDF
 */
template<typename T, int p>
T gaussianProbMul(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x) {
    const T sqrtDet = sqrt(cov.determinant());
    const Eigen::Matrix<T, p, 1> dif = x - mean;
    const T lhs = 1.0 / (pow(2.0 * M_PI, p / 2.0) * sqrtDet);
    const T rhs = exp(-0.5 * ((dif.transpose() * cov.inverse() * dif))(0, 0));
    return lhs * rhs;
}

/**
 * @brief compute an error ellipse
 * 
 * %   0.001 	0.002 	0.005 	0.01 	0.02 	0.05 	0.1 	0.2 	0.5 	0.75 	0.8 	0.9 	0.95 	0.98 	0.99 	0.995 	0.998 	0.999
 * kai 0.002 	0.004 	0.010 	0.020 	0.040 	0.103 	0.211 	0.446 	1.386 	2.773 	3.219 	4.605 	5.991 	7.824 	9.210 	10.597 	12.429 	13.816 
 * @param cov 
 * @param kai 
 * @return error [x, y, yaw]
 */
template<typename T, typename Matrix>
Eigen::Matrix<T, 3, 1> errorEllipse(const Matrix& cov, double kai = 5.911) {
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, 2, 2>> solver(cov);

    double l1 = solver.eigenvalues().x();
    double l2 = solver.eigenvalues().y();
    Eigen::Vector2d e1 = solver.eigenvectors().col(0);
    Eigen::Vector2d e2 = solver.eigenvectors().col(1);

    Eigen::Matrix<T, 3, 1> params;
    params[0] = std::sqrt(kai * l1);
    params[1] = std::sqrt(kai * l2);
    if (l2 >= l1)
        params[2] = std::atan2(e2.y(), e2.x());  
    else
        params[2] = std::atan2(e1.y(), e1.x());

    return params;
}

/**
 * @brief squared mahalanobis distance
 */
template<typename T, int p>
T squaredMahalanobisDistance(const Eigen::Matrix<T, p, 1>& mean, const Eigen::Matrix<T, p, p>& cov, const Eigen::Matrix<T, p, 1>& x){
    const Eigen::Matrix<T, p, 1> diff = x - mean;
    double distance = diff.transpose() * cov.inverse() * diff;
    return distance;
}

/**
 * @brief univariate squared mahalanobis distance
 */
template<typename T>
T squaredMahalanobisDistanceUni(T mean, T var, T x) {
    T diff = mean - x;
    return diff * diff / var;
}
}}