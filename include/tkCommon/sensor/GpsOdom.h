#pragma once
#include <sstream>
#include <fstream>
#include "tkCommon/utils.h"
#include "tkCommon/data/GpsData.h"

namespace tk{ namespace sensors{

    class GpsOdom {

    public:

        tk::common::Tfpose odom;
        tk::common::GeodeticConverter geoConv;

        GpsOdom() {
            odom = tk::common::Tfpose::Identity();
        }

        ~GpsOdom() {}

        void update(tk::data::GpsData &data) {
            //std::cout<<data<<"\n";

            if(data.header.stamp == 0 || data.lat == 0 || data.lon == 0)
                return;

            if(!geoConv.isInitialised()) {
                geoConv.initialiseReference(data.lat, data.lon, data.heigth);
                return;
            }

            double x, y, z;
            geoConv.geodetic2Enu(data.lat, data.lon, data.heigth, &x, &y, &z);
            odom = tk::common::odom2tf(x, y, z, 0, 0, 0);

            //std::cout<<tk::common::tf2pose(odom)<<" "<<tk::common::tf2rot(odom)<<"\n";
        }
    };
        
}}
