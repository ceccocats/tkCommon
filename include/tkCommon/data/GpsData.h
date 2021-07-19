#pragma once
#include "tkCommon/data/gen/GpsData_gen.h"

#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
#include <sensor_msgs/NavSatFix.h>
#endif
#if TKROS_VERSION == 2
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#endif
#endif


namespace tk { namespace data {

    class GpsData : public GpsData_gen {
    public:
#ifdef TKROS_ENABLED
#if TKROS_VERSION == 1
        void toRos(sensor_msgs::NavSatFix &msg) {
#endif
#if TKROS_VERSION == 2
        void toRos(sensor_msgs::msg::NavSatFix &msg) {
#endif
            this->header.toRos(msg.header);

            msg.latitude    = this->lat;
            msg.longitude   = this->lon;
            msg.altitude    = this->heigth;
            Eigen::Matrix3d e_cov = this->cov.matrix().transpose();
            memcpy(msg.position_covariance.data(), e_cov.data(), 3*3*sizeof(double)); 
        }
#if TKROS_VERSION == 1
        void fromRos(sensor_msgs::NavSatFix &msg) {
#endif
#if TKROS_VERSION == 2
        void fromRos(sensor_msgs::msg::NavSatFix &msg) {
#endif
            this->header.fromRos(msg.header);
            this->header.type   = DataType::GPS; 
            
            this->lat       = msg.latitude;
            this->lon       = msg.longitude; 
            this->heigth    = msg.altitude;
            this->sats      = 20;   // no sats data fill with good value
            Eigen::Matrix3d e_cov;
            memcpy(e_cov.data(), msg.position_covariance.data(), 3*3*sizeof(double)); 
            this->cov = e_cov.transpose();
        }
#endif
    };
}}
