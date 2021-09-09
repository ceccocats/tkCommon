#pragma once
#include "tkCommon/data/gen/ImuData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <sensor_msgs/Imu.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if TKROS_VERSION == 2
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#endif

namespace tk { namespace data {

    class ImuData : public ImuData_gen {
    public:
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(sensor_msgs::Imu &msg) {
#endif
#if TKROS_VERSION == 2
        void toRos(sensor_msgs::msg::Imu &msg) {
#endif
            this->header.toRos(msg.header);

            msg.linear_acceleration.x   = this->acc.x();
            msg.linear_acceleration.y   = this->acc.y();
            msg.linear_acceleration.z   = this->acc.z();
            memcpy(msg.linear_acceleration_covariance.data(), this->covAcc.data(), sizeof(double)*3*3);

            msg.angular_velocity.x  = this->angleVel.x();
            msg.angular_velocity.y  = this->angleVel.y();
            msg.angular_velocity.z  = this->angleVel.z();
            memcpy(msg.angular_velocity_covariance.data(), this->covAngleVel.data(), sizeof(double)*3*3);
            
            tf2::Quaternion q;
            q.setRPY(this->angle.x(), this->angle.y(), this->angle.z());
            q.normalize();
            msg.orientation = tf2::toMsg(q);
            memcpy(msg.orientation_covariance.data(), this->covAngle.data(), sizeof(double)*3*3);
        }

#if TKROS_VERSION == 1
        void fromRos(sensor_msgs::Imu &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(sensor_msgs::msg::Imu &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::IMU; 

            this->acc.x()   = msg.linear_acceleration.x;
            this->acc.y()   = msg.linear_acceleration.y;
            this->acc.z()   = msg.linear_acceleration.z; 
            memcpy(this->covAcc.data(), msg.linear_acceleration_covariance.data(), sizeof(double)*3*3);

            this->angleVel.x()  = msg.angular_velocity.x;
            this->angleVel.y()  = msg.angular_velocity.y;
            this->angleVel.z()  = msg.angular_velocity.z;
            memcpy(this->covAngleVel.data(), msg.angular_velocity_covariance.data(), sizeof(double)*3*3);

            Eigen::Quaterniond q;
            q.w()       = msg.orientation.w;
            q.x()       = msg.orientation.x;
            q.y()       = msg.orientation.y;
            q.z()       = msg.orientation.z;
            auto rpy    = q.toRotationMatrix().eulerAngles(0, 1, 2);

            this->angle.x() = rpy(0);
            this->angle.y() = rpy(1);
            this->angle.z() = rpy(2);
            memcpy(this->covAngle.data(), msg.orientation_covariance.data(), sizeof(double)*3*3);
        }

#endif
    };
}}
