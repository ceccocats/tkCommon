#pragma once
#include <tkCommon/common.h>

namespace tk { namespace common {

    struct XsensData_t {
        
    /**GPS data*////////////////////////
    double      latitude;
    double      longitude;
    double      altitude;
    timeStamp_t GPStimestamp;
    ////////////////////////////////////


    /**Mag X*///////////////////////////
    double      magX;
    double      magY;
    double      magZ;
    timeStamp_t MAGtimestamp;
    ////////////////////////////////////


    /**Accelerometer data*///////////////
    double      accX;
    double      accY;
    double      accZ;
    timeStamp_t ACCtimestamp;
    ////////////////////////////////////


    /**Gyroscope data */////////////////
    double      gyrX;
    double      gyrY;
    double      gyrZ;
    timeStamp_t GYRtimestamp;
    ////////////////////////////////////


    /**IMU data*////////////////////////
    double      yaw;
    double      pitch;
    double      roll;
    timeStamp_t IMUtimestamp;
    ////////////////////////////////////


    /**Environment*/////////////////////
    double      temperature;
    double      pressure;
    timeStamp_t ENVtimestamp;
    ////////////////////////////////////


    /**Velocity data*///////////////////
    double      velocity_x;
    double      velocity_y;
    double      velocity_z;
    timeStamp_t VELtimestamp;
    ////////////////////////////////////


    /**GNSS data*///////////////////////
    uint16_t    year;
    uint8_t     month;
    uint8_t     day;
    uint8_t     hour;
    uint8_t     min;
    uint8_t     sec;
    uint8_t     satellitiesNumber;
    uint16_t    geometricDOP;
    uint16_t    positionDOP;
    uint16_t    timeDOP;
    uint16_t    verticalDOP;
    uint16_t    horizzontalDOP;
    uint16_t    northingDOP;
    uint16_t    esatingDOP;
    timeStamp_t GNSStimestamp;
    ////////////////////////////////////


    /**
     * @brief Useful method for print struct in console
     * 
     * @param xsens Reference to XsensData_t struct
     */
    void consolePrint(){

        std::cout<<"TIME\t"<<(int)this->day<<"/"<<(int)this->month<<"/"<<(int)this->year<<"\t"<<(int)this->hour<<":"<<(int)this->min<<":"<<(int)this->sec<<"\n";
        std::cout<<"GPS\tlat: "<<this->latitude<<" lon: "<<this->longitude<<" alt: "<<this->altitude<<" timestamp: "<<this->GPStimestamp<<"\n";
        std::cout<<"MAG\tx: "<<this->magX<<" y: "<<this->magY<<" z: "<<this->magZ<<" timestamp: "<<this->MAGtimestamp<<"\n";
        std::cout<<"ACC\tx: "<<this->accX<<" y: "<<this->accY<<" z: "<<this->accZ<<" timestamp: "<<this->ACCtimestamp<<"\n";
        std::cout<<"GYR\tx: "<<this->gyrX<<" y: "<<this->gyrY<<" z: "<<this->gyrZ<<" timestamp: "<<this->GYRtimestamp<<"\n";
        std::cout<<"IMU\tyaw: "<<this->yaw<<" pitch: "<<this->pitch<<" roll: "<<this->roll<<" timestamp: "<<this->IMUtimestamp<<"\n";
        std::cout<<"ENV\ttemperature: "<<this->temperature<<" pressure: "<<this->pressure<<" timestamp: "<<this->ENVtimestamp<<"\n";
        std::cout<<"VEL\tx: "<<this->velocity_x<<" y: "<<this->velocity_y<<" z: "<<this->velocity_z<<" timestamp: "<<this->VELtimestamp<<"\n";
        std::cout<<"GNSS\tsat: "<<(int)this->satellitiesNumber<<" gDOP: "<<this->geometricDOP<<" pDOP: "<<this->positionDOP<<" tDOP: "<<this->timeDOP
                 <<" vDOP: "<<this->verticalDOP<<" hDOP: "<<this->horizzontalDOP<<" nDOP: "<<this->northingDOP<<" eDOP: "<<this->esatingDOP<<" timestamp: "<<this->GNSStimestamp<<"\n";
    }

    /**
     * @brief Useful method for clear all data struct
     * 
     * @param xsens Reference XsensData_t to struct
     */
    void clearAll(){

        this->latitude          = 0;
        this->longitude         = 0;
        this->altitude          = 0;
        this->GPStimestamp      = 0;
        this->accX              = 0;
        this->accY              = 0;
        this->accZ              = 0;
        this->ACCtimestamp      = 0;
        this->gyrX              = 0;
        this->gyrY              = 0;
        this->gyrZ              = 0;
        this->GYRtimestamp      = 0;
        this->yaw               = 0;
        this->pitch             = 0;
        this->roll              = 0;
        this->IMUtimestamp      = 0;
        this->temperature       = 0;
        this->pressure          = 0;
        this->ENVtimestamp      = 0;
        this->velocity_x        = 0;
        this->velocity_y        = 0;
        this->velocity_z        = 0;
        this->VELtimestamp      = 0;
        this->year              = 0;
        this->month             = 0;
        this->day               = 0;
        this->hour              = 0;
        this->min               = 0;
        this->sec               = 0;
        this->satellitiesNumber = 0;
        this->geometricDOP      = 0;
        this->positionDOP       = 0;
        this->timeDOP           = 0;
        this->verticalDOP       = 0;
        this->horizzontalDOP    = 0;
        this->northingDOP       = 0;
        this->esatingDOP        = 0;
        this->GNSStimestamp     = 0;
    }

    };
}
}