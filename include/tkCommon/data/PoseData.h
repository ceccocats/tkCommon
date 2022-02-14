#pragma once
#include "tkCommon/data/gen/PoseData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <geometry_msgs/PoseStamped.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#if TKROS_VERSION == 2
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#endif
#endif

namespace tk { namespace data {

    class PoseData : public PoseData_gen {
    public:
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(geometry_msgs::PoseStamped &msg) const {
#endif
#if TKROS_VERSION == 2
        void toRos(geometry_msgs::msg::PoseStamped &msg) const {
#endif
            this->header.toRos(msg.header);

            msg.pose.position.x = this->position.x();
            msg.pose.position.y = this->position.y();
            msg.pose.position.z = this->position.z();

            tf2::Quaternion q;
            q.setRPY(this->orientation.x(), this->orientation.y(), this->orientation.z());
            q.normalize();
            msg.pose.orientation = tf2::toMsg(q);
        }
#if TKROS_VERSION == 1
        void fromRos(const geometry_msgs::PoseStamped &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(const geometry_msgs::msg::PoseStamped &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::POSE; 
            
            this->position.x() = msg.pose.position.x;
            this->position.y() = msg.pose.position.y;
            this->position.z() = msg.pose.position.z;

            Eigen::Quaterniond q;
            q.w()       = msg.pose.orientation.w;
            q.x()       = msg.pose.orientation.x;
            q.y()       = msg.pose.orientation.y;
            q.z()       = msg.pose.orientation.z;
            auto rpy    = q.toRotationMatrix().eulerAngles(0, 1, 2);

            this->orientation.x() = rpy(0);
            this->orientation.y() = rpy(1);
            this->orientation.z() = rpy(2);
        }
#endif
    };
}}