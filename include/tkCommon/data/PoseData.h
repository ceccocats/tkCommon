#pragma once
#include "tkCommon/data/gen/PoseData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <geometry_msgs/PoseStamped.h>
#endif
#if TKROS_VERSION == 2
#include <std_msgs/msg/header.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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
        }
#if TKROS_VERSION == 1
        void fromRos(const geometry_msgs::PoseStamped &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(const geometry_msgs::msg::PoseStamped &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::POSE; 
        }
#endif
    };
}}