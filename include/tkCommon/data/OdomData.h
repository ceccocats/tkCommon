#pragma once
#include "tkCommon/data/gen/OdomData_gen.h"

#ifdef ROS_ENABLED
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif

namespace tk { namespace data {

    class OdomData : public OdomData_gen
    { 
    public:
#ifdef ROS_ENABLED
        void toRos(nav_msgs::Odometry &msg) {
            this->header.toRos(msg.header);

            msg.pose.pose.position.x    = this->pose.x();
            msg.pose.pose.position.y    = this->pose.y();
            msg.pose.pose.position.z    = this->pose.z();

            tf2::Quaternion q;
            q.setRPY(this->angle.x(), this->angle.y(), this->angle.z());
            q.normalize();
            msg.pose.pose.orientation   = tf2::toMsg(q);

            msg.twist.twist.linear.x    = speed;
        }

        void fromRos(nav_msgs::Odometry &msg) {
            this->header.fromRos(msg.header);

            this->pose.x()  = msg.pose.pose.position.x;
            this->pose.y()  = msg.pose.pose.position.y;
            this->pose.z()  = msg.pose.pose.position.z;

            Eigen::Quaterniond q;
            q.w()       = msg.pose.pose.orientation.w;
            q.x()       = msg.pose.pose.orientation.x;
            q.y()       = msg.pose.pose.orientation.y;
            q.z()       = msg.pose.pose.orientation.z;
            auto rpy    = q.toRotationMatrix().eulerAngles(0, 1, 2);

            this->angle.x()  = rpy(0);
            this->angle.y()  = rpy(1);
            this->angle.z()  = rpy(2);

            this->speed = std::sqrt(std::pow(msg.twist.twist.linear.x, 2) + std::pow(msg.twist.twist.linear.y, 2) + std::pow(msg.twist.twist.linear.z, 2)); 
        }
#endif
    };


}}
