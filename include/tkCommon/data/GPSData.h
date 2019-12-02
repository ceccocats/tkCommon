#pragma once

#include <tkCommon/data/DataHeader.h>

namespace tk { namespace data {

    /**
     *  GPS message frame definition
     */
    struct GPSData_t {
        tk::data::DataHeader_t header;

        double lat = 0, lon = 0, hdop =0, height = 0, age = 0;
        double quality = 0, sats = 0;

        // IMU
        double angleX = 0, angleY = 0, angleZ = 0;
        double angleRateX = 0, angleRateY = 0, angleRateZ = 0;
        double accX = 0, accY = 0, accZ = 0;
        double sideSlip = 0;

        friend std::ostream& operator<<(std::ostream& os, const GPSData_t& m) {
            os << m.header.stamp<< " Lat/Lon: " << m.lat <<"°/" << m.lon
               <<"°, Height: "<<m.height<<" Nsats: "<<m.sats;
            return os;
        }

        void init(){
            //TODO:
        }

        static const int nfields = 15;
        const char *fields[nfields] = {"stamp", "lat", "lon", "height", "sats", "angleX", "angleY", "angleZ",
                                   "angleRateX", "angleRateY", "angleRateZ", "accX", "accY", "accZ", "sideSlip"};
        double *vals[nfields-1] = {&lat, &lon, &height, &sats, &angleX, &angleY, &angleZ,
                                   &angleRateX, &angleRateY, &angleRateZ, &accX, &accY, &accZ, &sideSlip};

        matvar_t *toMatVar(std::string name = "gps") {

            const int n = nfields-1;

            size_t dim[2] = { 1, 1 }; // create 1x1 struct
            matvar_t* matstruct = Mat_VarCreateStruct(name.c_str(), 2, dim, fields, n+1); //main struct: Data

            matvar_t *var = Mat_VarCreate("stamp", MAT_C_UINT64, MAT_T_UINT64, 2, dim, &header.stamp, 0);
            Mat_VarSetStructFieldByName(matstruct, "stamp", 0, var); //0 for first row
            for(int i =0; i<n; i++) {
                matvar_t *var;
                var = Mat_VarCreate(fields[i+1], MAT_C_DOUBLE, MAT_T_DOUBLE, 2, dim, &vals[i], 0);
                Mat_VarSetStructFieldByName(matstruct, fields[i+1], 0, var); //0 for first row
            }
            return matstruct;
        }
        bool fromMatVar(matvar_t *var) {

            matvar_t *pvar = Mat_VarGetStructFieldByName(var, fields[0], 0);
            tkASSERT(pvar->class_type == MAT_C_UINT64);
            memcpy(&header.stamp, pvar->data, sizeof(uint64_t));

            for (int i = 1; i < nfields; i++) {
                matvar_t *pvar = Mat_VarGetStructFieldByName(var, fields[i], 0);
                tkASSERT(pvar->class_type == MAT_C_DOUBLE);
                memcpy(vals[i], pvar->data, sizeof(double));
            }
        }
    };
}
}