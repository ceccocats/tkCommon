#pragma once
#include "tkCommon/data/gen/OdomData_gen.h"

#ifdef ROS_ENABLED
#include <nav_msgs/Odometry.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#ifdef ROS2_ENABLED
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
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
            speed = s.speed;
            mode = s.mode;
        }
#ifdef ROS_ENABLED
        void toRos(nav_msgs::Odometry &msg) {
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

            msg.twist.twist.linear.x    = speed.x();
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

            mode = EULER;

            this->speed.x() = std::sqrt(std::pow(msg.twist.twist.linear.x, 2) + std::pow(msg.twist.twist.linear.y, 2) + std::pow(msg.twist.twist.linear.z, 2)); 
        }
#endif
#ifdef ROS2_ENABLED
        void toRos(nav_msgs::msg::Odometry &msg) {
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

            msg.twist.twist.linear.x    = speed.x();
        }

        void fromRos(nav_msgs::msg::Odometry &msg) {
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

            this->speed.x() = std::sqrt(std::pow(msg.twist.twist.linear.x, 2) + std::pow(msg.twist.twist.linear.y, 2) + std::pow(msg.twist.twist.linear.z, 2)); 
        }
#endif


    };


}}
