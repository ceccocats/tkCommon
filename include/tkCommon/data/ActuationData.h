#pragma once
#include "tkCommon/data/gen/ActuationData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include "ackermann_msgs/AckermannDriveStamped.h"
#elif TKROS_VERSION == 2
#include "ackermann_msgs/msg/ackermann_drive_stamped.hpp"
#endif
#endif

namespace tk { namespace data {

    class ActuationData : public ActuationData_gen{
        public:

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
            void toRos(ackermann_msgs::AckermannDriveStamped &msg)
#elif TKROS_VERSION == 2
            void toRos(ackermann_msgs::msg::AckermannDriveStamped &msg)
#endif
            {
                this->header.toRos(msg.header);
                msg.drive.steering_angle = steerAngle;
                msg.drive.acceleration = accel;
                msg.drive.speed = speed;
            }

#if TKROS_VERSION == 1
            void fromRos(ackermann_msgs::AckermannDriveStamped &msg)
#elif TKROS_VERSION == 2
            void fromRos(ackermann_msgs::msg::AckermannDriveStamped &msg)
#endif
            {
                this->header.fromRos(msg.header);
                this->header.type   = DataType::IMU; 
                steerAngle = msg.drive.steering_angle;
                accel = msg.drive.acceleration;
                speed = msg.drive.speed;
            }
#endif        
    };

}}
