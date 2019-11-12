#pragma once

#include "tkCommon/data/DataHeader.h"
#include <tkCommon/common.h>

namespace tk { namespace data {

    /**
     *  Vehicle data definition
     */
    struct VehicleData_t {

        tk::data::DataHeader_t header;

        double CAR_WHEELBASE;                   // car wheelbase [m]
        double CAR_DIM_X, CAR_DIM_Y, CAR_DIM_Z; // car dimensions [m]
        double CAR_BACK2AXLE;                   // from back to axle [m]
        double CAR_MASS;                        // car weight [kg]
        double CAR_FRONTAXLE_W;                 // width of front axle [m]
        double CAR_BACKAXLE_W;                  // width of back axle [m]
        double CAR_WHEEL_R;                     // wheel radius [m]

        double     speed    = 0;  // speed                [ m/s   ]
        double     speedKMH = 0;  // speed                [ Km/h  ]

        double     yawRate = 0;        // yaw rate             [ rad/s ]
        double     accelX  = 0;        // longitudinal acc     [ m/s2  ]
        double     accelY  = 0;        // lateral acc          [ m/s2  ]

        double     steerAngle     = 0;    // steering wheel angle [ rad   ]
        double     steerAngleRate = 0;    // steering wheel angle [ rad/s ]
        double     wheelAngle     = 0;    // wheel angle          [ rad   ]

        uint8_t    brakePedalSts        = 0;    // brake status   [ on/off ]
        double     brakeMasterPressure  = 0;    // brake pressure [ bar ]
        double     engineTorque         = 0;    // engine torque  [ % ]
        double     engineFrictionTorque = 0;    // friction torque[ % ]

        uint8_t    actualGear = 0;    // current Gear          [ int ]
        uint16_t   RPM = 0;           // RPM

        // wheel speeds [ m/s ]
        double     wheelFLspeed =0, wheelFRspeed =0, wheelRLspeed =0, wheelRRspeed =0;

        double     sideSlip = 0;      // beta angle            [ rad   ]
        uint8_t    tractionGrip = 0;  // traction control grip [ ? ]


        /**
         * Odometry vales
         */
        // x e y position of car in space
        long double x = 0;
        long double y = 0;

        // Direction of car in space
        long double carDirection = 0;

        // Boolean variable for know if new data is arrived
        bool Bspeed = 0, Bsteer = 0;
        int  posBuffer = 0;

        #define ODOM_BUFFER_SIZE 10
        struct odom_t {
            long double x = 0;
            long double y = 0;
            long double yaw = 0;
            uint64_t t = 0;
        } odometryBuffer[ODOM_BUFFER_SIZE];

        bool carOdometry(odom_t &odom, uint64_t time = 0){

            int bufPos = posBuffer;
            for(int i=1; i<ODOM_BUFFER_SIZE; i++) {
                int c = (bufPos - i) % ODOM_BUFFER_SIZE;
                if(c <0)
                    c += ODOM_BUFFER_SIZE;

                if(odometryBuffer[c].t < time || time == 0){
                    odom = odometryBuffer[c];
                    //if(odom.t == 0)
                    //    return false;
                    return true;
                }
            }
            return false;
        }
    };
}
}