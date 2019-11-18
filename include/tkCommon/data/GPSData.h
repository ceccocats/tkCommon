#pragma once

#include <tkCommon/data/DataHeader.h>

namespace tk { namespace data {

    /**
     *  GPS message frame definition
     */
    struct GPSData_t {
        tk::data::DataHeader_t header;

        timeStamp_t stamp = 0;

        double lat = 0, lon = 0, hdop =0, height = 0, age = 0;
        int quality = 0, sats = 0;

        // IMU
        double angleX, angleY, angleZ;
        double angleRateX, angleRateY, angleRateZ;
        double accX, accY, accZ;
        double sideSlip;

        friend std::ostream& operator<<(std::ostream& os, const GPSData_t& m) {
            os << "Lat/Lon: " << m.lat <<"°/" << m.lon
               <<"°, Height: "<<m.height<<" Nsats: "<<m.sats;
            return os;
        }

        void init(){
            //TODO:
        }
    };
}
}