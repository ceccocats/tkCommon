#pragma once
#include "tkCommon/data/gen/ImuData_gen.h"

#ifdef ROS_ENABLED
    #include <sensor_msgs/Imu.h>
#endif

namespace tk { namespace data {

    class ImuData : public ImuData_gen {
   
    public:

        #ifdef ROS_ENABLED
        bool toMsg(sensor_msgs::Imu &imu) {
            ros::Time ts;
            std_msgs::Header header;
            header.frame_id = "imu";
            header.stamp = ts.fromNSec(this->header.stamp*1e3);

            imu.header = header;
            imu.linear_acceleration.x = acc.x();
            imu.linear_acceleration.y = acc.y();
            imu.linear_acceleration.z = acc.z();
            imu.angular_velocity.x = angleVel.x();
            imu.angular_velocity.y = angleVel.y();
            imu.angular_velocity.z = angleVel.z();

            Eigen::Quaternionf q;
            q = Eigen::AngleAxisf(angle.x(), Eigen::Vector3f::UnitX())
              * Eigen::AngleAxisf(angle.y(), Eigen::Vector3f::UnitY())
              * Eigen::AngleAxisf(angle.z(), Eigen::Vector3f::UnitZ());
            imu.orientation.x = q.x();
            imu.orientation.y = q.y();
            imu.orientation.z = q.z();
            imu.orientation.w = q.w();
        }
        #endif

    };
}}