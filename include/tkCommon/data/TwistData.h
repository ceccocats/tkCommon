#pragma once
#include "tkCommon/data/gen/TwistData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <geometry_msgs/TwistStamped.h>
#include <geometry_msgs/Twist.h>
#endif
#if TKROS_VERSION == 2
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#endif
#endif

namespace tk { namespace data {

    class TwistData : public TwistData_gen {
    public:
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(geometry_msgs::TwistStamped &msg) const {
#endif
#if TKROS_VERSION == 2
        void toRos(geometry_msgs::msg::TwistStamped &msg) const {
#endif
            this->header.toRos(msg.header);
            msg.twist.linear.x = this->linear.x();
            msg.twist.linear.y = this->linear.y();
            msg.twist.linear.z = this->linear.z();

            msg.twist.angular.x = this->angular.x();
            msg.twist.angular.y = this->angular.y();
            msg.twist.angular.z = this->angular.z();
        }
#if TKROS_VERSION == 1
        void toRos(geometry_msgs::Twist &msg) const {
#endif
#if TKROS_VERSION == 2
        void toRos(geometry_msgs::msg::Twist &msg) const {
#endif
            msg.linear.x = this->linear.x();
            msg.linear.y = this->linear.y();
            msg.linear.z = this->linear.z();

            msg.angular.x = this->angular.x();
            msg.angular.y = this->angular.y();
            msg.angular.z = this->angular.z();
        }
#if TKROS_VERSION == 1
        void fromRos(const geometry_msgs::TwistStamped &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(const geometry_msgs::msg::TwistStamped &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::TWIST; 

            this->linear.x() = msg.twist.linear.x;
            this->linear.y() = msg.twist.linear.y;
            this->linear.z() = msg.twist.linear.z;

            this->angular.x() = msg.twist.angular.x;
            this->angular.y() = msg.twist.angular.y;
            this->angular.z() = msg.twist.angular.z;
        }
#endif
    };
}}
