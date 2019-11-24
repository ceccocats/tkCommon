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
        double angleX = 0, angleY = 0, angleZ = 0;
        double angleRateX = 0, angleRateY = 0, angleRateZ = 0;
        double accX = 0, accY = 0, accZ = 0;
        double sideSlip = 0;

        friend std::ostream& operator<<(std::ostream& os, const GPSData_t& m) {
            os << "Lat/Lon: " << m.lat <<"°/" << m.lon
               <<"°, Height: "<<m.height<<" Nsats: "<<m.sats;
            return os;
        }

        void init(){
            //TODO:
        }

        matvar_t *matVar(std::string name = "gps") {

            const int n = 14;
            const char *fields[n+1] = {"stamp", "lat", "lon", "height", "sats", "angleX", "angleY", "angleZ",
                                    "angleRateX", "angleRateY", "angleRateZ", "accX", "accY", "accZ", "sideSlip"};
            double vals[n] = {lat, lon, height, (double)sats, angleX, angleY, angleZ,
                               angleRateX, angleRateY, angleRateZ, accX, accY, accZ, sideSlip};

            size_t dim[2] = { 1, 1 }; // create 1x1 struct
            matvar_t* matstruct = Mat_VarCreateStruct(name.c_str(), 2, dim, fields, n); //main struct: Data

            matvar_t *var = Mat_VarCreate("stamp", MAT_C_UINT64, MAT_T_UINT64, 2, dim, &stamp, 0);
            Mat_VarSetStructFieldByName(matstruct, "stamp", 0, var); //0 for first row
            for(int i =0; i<n; i++) {
                matvar_t *var;
                var = Mat_VarCreate(fields[i+1], MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dim, &vals[i], 0);
                Mat_VarSetStructFieldByName(matstruct, fields[i+1], 0, var); //0 for first row
            }
            return matstruct;
        }
    };
}
}