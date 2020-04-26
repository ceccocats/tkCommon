#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/math/MatIO.h"

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

        /**
         *
         */
        void init() override {
            SensorData::init();

            header.sensor = sensorName::GPS;

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
            os << std::setprecision(10) << m.header.stamp<< " Lat/Lon: " << m.lat <<"°/" << m.lon
               <<"°, Height: "<<m.height<<" Nsats: "<<m.sats<<" quality: "<<m.quality;
            return os;
        }


        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            tk::math::MatIO::var_t hvar;
            tk::data::SensorData::toVar("header", hvar);

            std::vector<tk::math::MatIO::var_t> structVars(15);
            structVars[ 0] = hvar;
            structVars[ 1].set("lat",        lat);
            structVars[ 2].set("lon",        lon);
            structVars[ 3].set("height",     height);
            structVars[ 4].set("sats",       sats);
            structVars[ 5].set("angleX",     angleX);
            structVars[ 6].set("angleY",     angleY);
            structVars[ 7].set("angleZ",     angleZ);
            structVars[ 8].set("angleRateX", angleRateX);
            structVars[ 9].set("angleRateY", angleRateY);
            structVars[10].set("angleRateZ", angleRateZ);
            structVars[11].set("accX",       accX);
            structVars[12].set("accY",       accY);
            structVars[13].set("accZ",       accZ);
            structVars[14].set("sideSlip",   sideSlip);
            return var.setStruct(name, structVars);
        }


        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;

            tk::data::SensorData::fromVar(var["header"]);
            var["lat"       ].get(lat);
            var["lon"       ].get(lon);
            var["height"    ].get(height);
            var["sats"      ].get(sats);
            var["angleX"    ].get(angleX);
            var["angleY"    ].get(angleY);
            var["angleZ"    ].get(angleZ);
            var["angleRateX"].get(angleRateX);
            var["angleRateY"].get(angleRateY);
            var["angleRateZ"].get(angleRateZ);
            var["accX"      ].get(accX);
            var["accY"      ].get(accY);
            var["accZ"      ].get(accZ);
            var["sideSlip"  ].get(sideSlip);
            return true;
        }
    };
}}