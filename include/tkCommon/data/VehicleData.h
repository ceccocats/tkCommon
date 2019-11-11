#pragma once
#include <tkCommon/common.h>

namespace tk { namespace data {

    /**
     *  Vehicle data definition
     */
    struct VehicleData_t {

        tk::data::DataHeader_t header;

        int    CAN_TYPE; // 0 = CCAN, 1 = PWTCAN
        double CAR_WHEELBASE;                   // car wheelbase [m]
        double CAR_DIM_X, CAR_DIM_Y, CAR_DIM_Z; // car dimensions [m]
        double CAR_BACK2AXLE;                   // from back to axle [m]
        double CAR_MASS;                        // car weight [kg]
        double CAR_FRONTAXLE_W;                 // width of front axle [m]
        double CAR_BACKAXLE_W;                  // width of back axle [m]
        double CAR_WHEEL_R;                     // wheel radius [m]

        double     speed;         // speed                [ m/s   ]
        double     speedKMH;      // speed                [ Km/h  ]

        double     yawRate;       // yaw rate             [ rad/s ]
        double     accelX;        // longitudinal acc     [ m/s2  ]
        double     accelY;        // lateral acc          [ m/s2  ]

        double     steerAngle;            // steering wheel angle [ rad   ]
        double     steerAngleRate;        // steering wheel angle [ rad/s ]
        double     wheelAngle;            // wheel angle          [ rad   ]

        uint8_t    brakePedalSts;         // brake status   [ on/off ]
        double     brakeMasterPressure;   // brake pressure [ bar ]
        double     engineTorque;          // engine torque  [ % ]
        double     engineFrictionTorque;  // friction torque[ % ]

        uint8_t    actualGear;    // current Gear          [ int ]
        uint16_t   RPM;           // RPM

        // wheel speeds [ m/s ]
        double     wheelFLspeed, wheelFRspeed, wheelRLspeed, wheelRRspeed;

        double     sideSlip;      // beta angle            [ rad   ]
        uint8_t    tractionGrip;  // traction control grip [ ? ]
    };
}
}