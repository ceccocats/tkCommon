#pragma once
#include "tkCommon/data/gen/ImuData_gen.h"

#ifdef ROS_ENABLED
#include <sensor_msgs/Imu.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#ifdef ROS2_ENABLED
#include <sensor_msgs//msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif

namespace tk { namespace data {

    class ImuData : public ImuData_gen {
    public:
#ifdef ROS_ENABLED
        void toRos(sensor_msgs::Imu &msg) {
            this->header.toRos(msg.header);

            msg.linear_acceleration.x   = this->acc.x();
            msg.linear_acceleration.y   = this->acc.y();
            msg.linear_acceleration.z   = this->acc.z();

            msg.angular_velocity.x  = this->angleVel.x();
            msg.angular_velocity.y  = this->angleVel.y();
            msg.angular_velocity.z  = this->angleVel.z();
            
            tf2::Quaternion q;
            q.setRPY(this->angle.x(), this->angle.y(), this->angle.z());
            q.normalize();
            msg.orientation = tf2::toMsg(q);
        }

        void fromRos(sensor_msgs::Imu &msg) {
            this->header.fromRos(msg.header);
            this->header.type   = DataType::IMU; 

            this->acc.x()   = msg.linear_acceleration.x;
            this->acc.y()   = msg.linear_acceleration.y;
            this->acc.z()   = msg.linear_acceleration.z; 

            this->angleVel.x()  = msg.angular_velocity.x;
            this->angleVel.y()  = msg.angular_velocity.y;
            this->angleVel.z()  = msg.angular_velocity.z;

            Eigen::Quaterniond q;
            q.w()       = msg.orientation.w;
            q.x()       = msg.orientation.x;
            q.y()       = msg.orientation.y;
            q.z()       = msg.orientation.z;
            auto rpy    = q.toRotationMatrix().eulerAngles(0, 1, 2);

            this->angle.x() = rpy(0);
            this->angle.y() = rpy(1);
            this->angle.z() = rpy(2);
        }
#endif
    };
}}
