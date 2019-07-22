#pragma once
#include <tkCommon/common.h>

namespace tk { namespace common {

    struct XsensData_t {
        
    /**GPS data*////////////////////////
    float64_t   latitude;
    float64_t   longitude;
    float64_t   altitude;
    timeStamp_t GPStimestamp;
    ////////////////////////////////////


    /**Mag X*///////////////////////////
    float64_t   magX;
    float64_t   magY;
    float64_t   magZ;
    timeStamp_t MAGtimestamp;
    ////////////////////////////////////


    /**Accelerometer data*///////////////
    float64_t   accX;
    float64_t   accY;
    float64_t   accZ;
    timeStamp_t ACCtimestamp;
    ////////////////////////////////////


    /**Gyroscope data */////////////////
    float64_t   gyrX;
    float64_t   gyrY;
    float64_t   gyrZ;
    timeStamp_t GYRtimestamp;
    ////////////////////////////////////


    /**IMU data*////////////////////////
    float64_t   yaw;
    float64_t   pitch;
    float64_t   roll;
    timeStamp_t IMUtimestamp;
    ////////////////////////////////////


    /**Environment*/////////////////////
    float64_t   temperature;
    float64_t   pressure;
    timeStamp_t ENVtimestamp;
    ////////////////////////////////////


    /**Velocity data*///////////////////
    float64_t   velocity_x;
    float64_t   velocity_y;
    float64_t   velocity_z;
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
    void consolePrint(XsensData_t& xsens){

        std::cout<<"TIME\t"<<xsens.day"/"<<xsens.month<<"/"<<xsens.year<<"\t"<<xsens.hour<<":"<<xsens.min<<":"<<xsens.sec<<"\n";
        std::cout<<"GPS\tlat: "<<xsens.latitude<<" lon: "<<xsens.longitude<<" alt: "<<xsens.altitude<<" timestamp: "<<xsens.GPStimestamp<<"\n";
        std::cout<<"MAG\tx: "<<xsens.magX<<" y: "<<xsens.magY<<" z: "<<xsens.magZ<<" timestamp: "<<xsens.MAGtimestamp<<"\n";
        std::cout<<"ACC\tx: "<<xsens.accX<<" y: "<<xsens.accY<<" z: "<<xsens.accZ<<" timestamp: "<<xsens.ACCtimestamp<<"\n";
        std::cout<<"GYR\tx: "<<xsens.gyrX<<" y: "<<xsens.gyrY<<" z: "<<xsens.gyrZ<<" timestamp: "<<xsens.GYRtimestamp<<"\n";
        std::cout<<"IMU\tyaw: "<<xsens.yaw<<" pitch: "<<xsens.pitch<<" roll: "<<xsens.roll<<" timestamp: "<<xsens.IMUtimestamp<<"\n";
        std::cout<<"ENV\ttemperature: "<<xsens.temperature<<" pressure: "<<xsens.pressure<<" timestamp: "<<xsens.ENVtimestamp<<"\n";
        std::cout<<"VEL\tx: "<<xsens.velocity_x<<" y: "<<xsens.velocity_y<<" z: "<<xsens.velocity_z<<" timestamp: "<<xsens.VELtimestamp<<"\n";
        std::cout<<"GNSS\tsat: "<<xsens.satellitiesNumber<<" gDOP: "<<xsens.geometricDOP<<" pDOP: "<<xsens.positionDOP<<" tDOP: "<<xsens.timeDOP
                 <<" vDOP: "<<xsens.verticalDOP<<" hDOP: "<<xsens.horizzontalDOP<<" nDOP: "<<xsens.northingDOP<<" eDOP: "<<xsens.esatingDOP<<" timestamp: "<<xsens.GNSStimestamp<<"\n";
    }

    /**
     * @brief Useful method for clear all data struct
     * 
     * @param xsens Reference XsensData_t to struct
     */
    void clearAll(XsensData_t& xsens){

        xsens.latitude          = 0;
        xsens.longitude         = 0;
        xsens.altitude          = 0;
        xsens.GPStimestamp      = 0;
        xsens.accX              = 0;
        xsens.accY              = 0;
        xsens.accZ              = 0;
        xsens.ACCtimestamp      = 0;
        xsens.gyrX              = 0;
        xsens.gyrY              = 0;
        xsens.gyrZ              = 0;
        xsens.GYRtimestamp      = 0;
        xsens.yaw               = 0;
        xsens.pitch             = 0;
        xsens.roll              = 0;
        xsens.IMUtimestamp      = 0;
        xsens.temperature       = 0;
        xsens.pressure          = 0;
        xsens.ENVtimestamp      = 0;
        xsens.velocity_x        = 0;
        xsens.velocity_y        = 0;
        xsens.velocity_z        = 0;
        xsens.VELtimestamp      = 0;
        xsens.year              = 0;
        xsens.month             = 0;
        xsens.day               = 0;
        xsens.hour              = 0;
        xsens.min               = 0;
        xsens.sec               = 0;
        xsens.satellitiesNumber = 0;
        xsens.geometricDOP      = 0;
        xsens.positionDOP       = 0;
        xsens.timeDOP           = 0;
        xsens.verticalDOP       = 0;
        xsens.horizzontalDOP    = 0;
        xsens.northingDOP       = 0;
        xsens.esatingDOP        = 0;
        xsens.GNSStimestamp     = 0;
    }

    };
}
}