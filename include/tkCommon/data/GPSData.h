#pragma once

#include "tkCommon/data/SensorData.h"

namespace tk { namespace data {

    /**
     *  GPS message frame definition
     */
    class GPSData : public SensorData {
    public:
        // GPS
        double lat, lon, hdop, height;

        // stat
        double age, quality, sats;

        // IMU
        double angleX, angleY, angleZ;
        double angleRateX, angleRateY, angleRateZ;
        double accX, accY, accZ;
        double sideSlip;

        static const int GPS_FIELDS = 15;
        const char  *fields[GPS_FIELDS] = {"stamp", "lat", "lon", "height", "sats", "angleX", "angleY", "angleZ",
                                           "angleRateX", "angleRateY", "angleRateZ", "accX", "accY", "accZ", "sideSlip"};
        double      *vals[GPS_FIELDS-1] = {&lat, &lon, &height, &sats, &angleX, &angleY, &angleZ,
                                           &angleRateX, &angleRateY, &angleRateZ, &accX, &accY, &accZ, &sideSlip};

        /**
         *
         */
        void init() override {
            SensorData::init();

            // GPS
            lat     = 0;
            lon     = 0;
            hdop    = 0;
            height  = 0;

            // stat
            age     = 0;
            quality = 0;
            sats    = 0;

            // IMU
            angleX      = 0;
            angleY      = 0;
            angleZ      = 0;
            angleRateX  = 0;
            angleRateY  = 0;
            angleRateZ  = 0;
            accX        = 0;
            accY        = 0;
            accZ        = 0;
            sideSlip    = 0;
        }

        /**
         *
         */
        void release() override {}

        /**
         *
         * @param s
         * @return
         */
        bool checkDimension(SensorData *s) override {
            auto *source = dynamic_cast<GPSData *>(s);
            return true;
        }

        /**
         *
         * @param s
         * @return
         */
        GPSData& operator=(const GPSData& s) {
            SensorData::operator=(s);

            // GPS
            this->lat     = s.lat;
            this->lon     = s.lon;
            this->hdop    = s.hdop;
            this->height  = s.height;

            // stat
            this->age     = s.age;
            this->quality = s.quality;
            this->sats    = s.sats;

            // IMU
            this->angleX      = s.angleX;
            this->angleY      = s.angleY;
            this->angleZ      = s.angleZ;
            this->angleRateX  = s.angleRateX;
            this->angleRateY  = s.angleRateY;
            this->angleRateZ  = s.angleRateZ;
            this->accX        = s.accX;
            this->accY        = s.accY;
            this->accZ        = s.accZ;
            this->sideSlip    = s.sideSlip;

            return *this;
        }

        /**
         *
         * @param os
         * @param m
         * @return
         */
        friend std::ostream& operator<<(std::ostream& os, const GPSData& m) {
            os << m.header.stamp<< " Lat/Lon: " << m.lat <<"°/" << m.lon
               <<"°, Height: "<<m.height<<" Nsats: "<<m.sats;
            return os;
        }

        /**
         *
         * @param name
         * @return
         */
        matvar_t *toMatVar(std::string name = "gps") {

            const int n = GPS_FIELDS - 1;

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

        /**
         *
         * @param var
         * @return
         */
        bool fromMatVar(matvar_t *var) {

            matvar_t *pvar = Mat_VarGetStructFieldByName(var, fields[0], 0);
            tkASSERT(pvar->class_type == MAT_C_UINT64);
            memcpy(&header.stamp, pvar->data, sizeof(uint64_t));

            for (int i = 1; i < GPS_FIELDS; i++) {
                matvar_t *pvar = Mat_VarGetStructFieldByName(var, fields[i], 0);
                tkASSERT(pvar->class_type == MAT_C_DOUBLE);
                memcpy(vals[i], pvar->data, sizeof(double));
            }
        }
    };
}}