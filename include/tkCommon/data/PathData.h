#pragma once
#include "tkCommon/data/gen/PathData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <nav_msgs/Path.h>
#endif
#if TKROS_VERSION == 2
#include <std_msgs/msg/header.hpp>
#include <nav_msgs/msg/path.hpp>
#endif
#endif

namespace tk { namespace data {

    class PathData : public PathData_gen {
    public:
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(nav_msgs::Path &msg) const {
#endif
#if TKROS_VERSION == 2
        void toRos(nav_msgs::msg::Path &msg) const {
#endif
            this->header.toRos(msg.header);
            
            msg.poses.resize(this->poses.size());
            for (size_t i = 0; i < this->poses.size(); ++i)
                this->poses[i].toRos(msg.poses[i]);
        }
#if TKROS_VERSION == 1
        void fromRos(const nav_msgs::Path &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(const nav_msgs::msg::Path &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::PATH; 
            
            this->poses.resize(msg.poses.size());
            for (size_t i = 0; i < this->poses.size(); ++i)
                this->poses[i].fromRos(msg.poses[i]);
        }
#endif
    };
}}
