#include "tkCommon/math/Vec.h"

using namespace tk::math;

Vec3<double> 
tk::math::quat2euler(Vec4<double> aQuaternion) 
{
    Vec3<double> euler;

    // roll (x-axis rotation)
    double sinr_cosp = 2 * (aQuaternion.w() * aQuaternion.x() + aQuaternion.y() * aQuaternion.z());
    double cosr_cosp = 1 - 2 * (aQuaternion.x() * aQuaternion.x() + aQuaternion.y() * aQuaternion.y());
    euler.x() = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (aQuaternion.w() * aQuaternion.y() - aQuaternion.z() * aQuaternion.x());
    if (std::abs(sinp) >= 1)
        euler.y() = std::copysign(M_PI / 2, sinp); // use 90 degrees if out of range
    else
        euler.y() = std::asin(sinp);

    // yaw (z-axis rotation)
    double siny_cosp = 2 * (aQuaternion.w() * aQuaternion.z() + aQuaternion.x() * aQuaternion.y());
    double cosy_cosp = 1 - 2 * (aQuaternion.y() * aQuaternion.y() + aQuaternion.z() * aQuaternion.z());
    euler.z() = std::atan2(siny_cosp, cosy_cosp);

    return euler;
}

Vec4<double> 
tk::math::euler2quat(Vec3<double> aEuler) 
{
    // Abbreviations for the various angular functions
    double cy = std::cos(aEuler.z() * 0.5);
    double sy = std::sin(aEuler.z() * 0.5);
    double cp = std::cos(aEuler.y() * 0.5);
    double sp = std::sin(aEuler.y() * 0.5);
    double cr = std::cos(aEuler.x() * 0.5);
    double sr = std::sin(aEuler.x() * 0.5);

    Vec4<double>  q;
    q.w() = cr * cp * cy + sr * sp * sy;
    q.x() = sr * cp * cy - cr * sp * sy;
    q.y() = cr * sp * cy + sr * cp * sy;
    q.z() = cr * cp * sy - sr * sp * cy;

    return q;
}