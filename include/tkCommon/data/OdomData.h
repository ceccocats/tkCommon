#pragma once
#include "tkCommon/data/gen/OdomData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if TKROS_VERSION == 2
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#endif

namespace tk { namespace data {

    class OdomData : public OdomData_gen
    { 
    public:
        enum angleMode {EULER, QUATERNION};
        angleMode mode = EULER;
        OdomData(){};
        OdomData(const tk::data::OdomData& s){
            header.tf = s.header.tf;
            pose = s.pose;
            angle = s.angle;
            linear_velocity = s.linear_velocity;
            angular_velocity = s.angular_velocity;
            mode = s.mode;
        }

        float getLinearSpeed() {
            return std::sqrt(std::pow(linear_velocity.x(), 2) + std::pow(linear_velocity.y(), 2) + std::pow(linear_velocity.z(), 2));
        }
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(nav_msgs::Odometry &msg) const {
#endif
#if TKROS_VERSION == 2
        void toRos(nav_msgs::msg::Odometry &msg) const {
#endif
            this->header.toRos(msg.header);

            msg.pose.pose.position.x    = this->pose.x();
            msg.pose.pose.position.y    = this->pose.y();
            msg.pose.pose.position.z    = this->pose.z();

            tf2::Quaternion q;
            if(mode == EULER){
                q.setRPY(this->angle.x(), this->angle.y(), this->angle.z());
            }
            else{
                q.setX(angle.x());
                q.setY(angle.y());
                q.setZ(angle.z());
                q.setW(angle.w());
            }
            q.normalize();
            msg.pose.pose.orientation   = tf2::toMsg(q);

            msg.twist.twist.linear.x    = linear_velocity.x();
            msg.twist.twist.linear.y    = linear_velocity.y();
            msg.twist.twist.linear.z    = linear_velocity.z();

            msg.twist.twist.angular.x   = angular_velocity.x();
            msg.twist.twist.angular.y   = angular_velocity.y();
            msg.twist.twist.angular.z   = angular_velocity.z();
        }

#if TKROS_VERSION == 1
        void fromRos(const nav_msgs::Odometry &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(const nav_msgs::msg::Odometry &msg) {
#endif
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

            mode = EULER;

            linear_velocity.x()     = msg.twist.twist.linear.x; 
            linear_velocity.y()     = msg.twist.twist.linear.y; 
            linear_velocity.z()     = msg.twist.twist.linear.z; 

            angular_velocity.x()    = msg.twist.twist.angular.x;
            angular_velocity.y()    = msg.twist.twist.angular.y;
            angular_velocity.z()    = msg.twist.twist.angular.z;
        }
#endif
    };
}}
