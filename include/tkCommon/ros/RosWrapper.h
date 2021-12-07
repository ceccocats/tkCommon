#pragma once

#if TKROS_VERSION == 1
#include <ros/ros.h>
#include <tf2_msgs/TFMessage.h>
#elif TKROS_VERSION == 2
#include <rclcpp/rclcpp.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#endif

#include <tkCommon/log.h>

namespace tk { namespace ros_wrapper {
    class RosWrapper {
    private:
        bool initted = false;

    public:
        static RosWrapper instance;

#if TKROS_VERSION == 1
        ros::NodeHandle *n = nullptr;
#elif TKROS_VERSION == 2
        rclcpp::Node::SharedPtr n = nullptr;
#endif
        pthread_t rosSpinThread;

         RosWrapper() = default;
        ~RosWrapper() = default;

        bool init();
        void close();

        static void *spinThread(void*);
    };
}}
