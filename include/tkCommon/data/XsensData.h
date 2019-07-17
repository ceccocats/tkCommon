#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    struct XsensData_t {
        
    //Latitude
    double latitude;
    //Longitude
    double longitude;
    //Altitude
    double altitude;
    //GPS Timestamp
    timeStamp_t GPStimestamp;

    //Mag X
    double magX;
    //Mag y
    double magY;
    //Mag z
    double magZ;
    //Mag timestamp
    timeStamp_t MAGtimestamp;

    //Acc x
    double accX;
    //Acc y
    double accY;
    //Acc z
    double accZ;
    //Acc timestamp
    timeStamp_t ACCtimestamp;

    //Gyr x
    double gyrX;
    //Gyr y
    double gyrY;
    //Gyr z
    double gyrZ;
    //Gyr timestamp
    timeStamp_t GYRtimestamp;

    //Yaw
    double yaw;
    //Pitch
    double pitch;
    //Roll
    double roll;
    //IMU timestamp
    timeStamp_t IMUtimestamp;
    };
}
}