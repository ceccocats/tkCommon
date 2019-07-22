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

    void consolePrint(XsensData_t& xsens){

        std::cout<<"TIME\t"<<xsens.day"/"<<xsens.month<<"/"<<xsens.year<<"\t"<<xsens.hour<<":"<<xsens.min<<":"<<xsens.sec<<"\n";
        std::cout<<"GPS\tlat: "<<xsens.latitude<<" lon: "<<xsens.longitude<<" alt: "<<xsens.altitude<<" timestamp: "<<xsens.GPStimestamp<<"\n";
        std::cout<<"MAG\tx: "<<xsens.magX<<" y: "<<xsens.magY<<" z: "<<xsens.magZ<<" timestamp: "<<xsens.MAGtimestamp<<"\n";
        std::cout<<"ACC\tx: "<<xsens.accX<<" y: "<<xsens.accY<<" z: "<<xsens.accZ<<" timestamp: "<<xsens.ACCtimestamp<<"\n";
        std::cout<<"GYR\tx: "<<xsens.gyrX<<" y: "<<xsens.gyrY<<" z: "<<xsens.gyrZ<<" timestamp: "<<xsens.GYRtimestamp<<"\n";
        std::cout<<"IMU\tyaw: "<<xsens.yaw<<" pitch: "<<xsens.pitch<<" roll: "<<xsens.roll<<" timestamp: "<<xsens.IMUtimestamp<<"\n";
        std::cout<<"ENV\ttemperature: "<<xsens.temperature<<" pressure: "<<xsens.pressure<<"\n";
        std::cout<<"VEL\tx: "<<xsens.velocity_x<<" y: "<<xsens.velocity_y<<" z: "<<xsens.velocity_z<<" timestamp: "<<xsens.VELtimestamp<<"\n";
        std::cout<<"GNSS\tsat: "<<xsens.satellitiesNumber<<" gDOP: "<<xsens.geometricDOP<<" pDOP: "<<xsens.positionDOP<<" tDOP: "<<xsens.timeDOP
                 <<" vDOP: "<<xsens.verticalDOP<<" hDOP: "<<xsens.horizzontalDOP<<" nDOP: "<<xsens.northingDOP<<" eDOP: "<<xsens.esatingDOP<<" timestamp: "<<xsens.GNSStimestamp<<"\n";
    }

    };
}
}

latitude;
    double      longitude;
    double      altitude;
    timeStamp_t GPStimestamp;