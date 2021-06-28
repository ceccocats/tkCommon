#pragma once
#include "tkCommon/data/gen/ActuationData_gen.h"

#ifdef ROS_ENABLED
#include "ackermann_msgs/AckermannDriveStamped.h"
#endif

namespace tk { namespace data {

    class ActuationData : public ActuationData_gen{
        
#ifdef ROS_ENABLED
        void toRos(ackermann_msgs::AckermannDriveStamped &msg) {
            this->header.toRos(msg.header);
            msg.drive.steering_angle = steerAngle;
            msg.drive.acceleration = accel;
            msg.drive.speed = speed;
        }

        void fromRos(ackermann_msgs::AckermannDriveStamped &msg) {
            this->header.fromRos(msg.header);
            this->header.type   = DataType::IMU; 
            steerAngle = msg.drive.steering_angle;
            accel = msg.drive.acceleration;
            speed = msg.drive.speed;
        }
#endif        
    };

}}
