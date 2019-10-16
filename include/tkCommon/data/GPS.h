#pragma once
#include <tkCommon/common.h>

namespace tk { namespace common {

    struct GPS_t {
        
    /**GPS data*////////////////////////
    double      latitude;
    double      longitude;
    double      altitude;
    timeStamp_t gpsTimestamp;
    //
    bool        gpsFilled;
    ////////////////////////////////////


    /**Accelerometer data*///////////////
    double      accX;
    double      accY;
    double      accZ;
    timeStamp_t accTimestamp;
    //
    bool        accFilled;
    ////////////////////////////////////

//angle
//angleRate
//acc
//velocity

    /**Gyroscope data */////////////////
    double      gyrX;
    double      gyrY;
    double      gyrZ;
    timeStamp_t gyrTimestamp;
    //
    bool        gyrFilled;
    ////////////////////////////////////


    /**IMU data*////////////////////////angle
    double      yaw;
    double      pitch;
    double      roll;
    timeStamp_t imuTimestamp;
    //
    bool        imuFilled;
    ////////////////////////////////////


    /**Velocity data*///////////////////
    double      velocity_x;
    double      velocity_y;
    double      velocity_z;
    timeStamp_t VELtimestamp;
    //
    bool        velFilled;
    ////////////////////////////////////


    /**GNSS data*///////////////////////covarianza
    uint16_t    year;
    uint8_t     month;
    uint8_t     day;
    uint8_t     hour;
    uint8_t     min;
    uint8_t     sec;
    uint8_t     satellitiesNumber;
    uint16_t    positionDOP;
    uint16_t    horizzontalDOP;
    timeStamp_t gnssTimestamp;
    //
    bool        gnssFilled;
    ////////////////////////////////////


    /**
     * @brief Useful method for print struct in console
     * 
     * @param Reference to GPS_t struct
     */
    void consolePrint(){

        if(this->gnssFilled)
            std::cout<<"TIME\t"<<(int)this->day<<"/"<<(int)this->month<<"/"<<(int)this->year<<"\t"<<(int)this->hour<<":"<<(int)this->min<<":"<<(int)this->sec<<"\n";
        if(this->gpsFilled)
            std::cout<<"GPS\tlat: "<<this->latitude<<" lon: "<<this->longitude<<" alt: "<<this->altitude<<" timestamp: "<<this->gpsTimestamp<<"\n";
        if(this->magFilled)
            std::cout<<"MAG\tx: "<<this->magX<<" y: "<<this->magY<<" z: "<<this->magZ<<" timestamp: "<<this->magTimestamp<<"\n";
        if(this->accFilled)
            std::cout<<"ACC\tx: "<<this->accX<<" y: "<<this->accY<<" z: "<<this->accZ<<" timestamp: "<<this->accTimestamp<<"\n";
        if(this->gyrFilled)
            std::cout<<"GYR\tx: "<<this->gyrX<<" y: "<<this->gyrY<<" z: "<<this->gyrZ<<" timestamp: "<<this->gyrTimestamp<<"\n";
        if(this->imuFilled)
            std::cout<<"IMU\tyaw: "<<this->yaw<<" pitch: "<<this->pitch<<" roll: "<<this->roll<<" timestamp: "<<this->imuTimestamp<<"\n";
        if(this->velFilled)
            std::cout<<"VEL\tx: "<<this->velocity_x<<" y: "<<this->velocity_y<<" z: "<<this->velocity_z<<" timestamp: "<<this->vleTimestamp<<"\n";
        if(this->gnssFilled)
            std::cout<<"GNSS\tsat: "<<(int)this->satellitiesNumber<<" pDOP: "<<this->positionDOP<<" hDOP: "<<this->horizzontalDOP<<" timestamp: "<<this->gnssTimestamp<<"\n";
    }

    /**
     * @brief Useful method for clear all data struct
     * 
     * @param Reference GPS_t to struct
     */
    void clearAll(){

        this->latitude          = 0;
        this->longitude         = 0;
        this->altitude          = 0;
        this->gpsTimestamp      = 0;
        this->accX              = 0;
        this->accY              = 0;
        this->accZ              = 0;
        this->accTimestamp      = 0;
        this->gyrX              = 0;
        this->gyrY              = 0;
        this->gyrZ              = 0;
        this->gyrTimestamp      = 0;
        this->yaw               = 0;
        this->pitch             = 0;
        this->roll              = 0;
        this->imuTimestamp      = 0;
        this->velocity_x        = 0;
        this->velocity_y        = 0;
        this->velocity_z        = 0;
        this->velTimestamp      = 0;
        this->year              = 0;
        this->month             = 0;
        this->day               = 0;
        this->hour              = 0;
        this->min               = 0;
        this->sec               = 0;
        this->satellitiesNumber = 0;
        this->positionDOP       = 0;
        this->horizzontalDOP    = 0;
        this->gnssTimestamp     = 0;
    }

    };
}
}