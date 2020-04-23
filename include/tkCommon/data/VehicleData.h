#pragma once
#include "tkCommon/data/SensorData.h"
#include "tkCommon/common.h"
#include "tkCommon/math/MatIO.h"

namespace tk { namespace data {

    /**
     *  Vehicle data definition
     */
    class VehicleData : public SensorData {
        
        public:

        double CAR_WHEELBASE;                   /** car wheelbase        [m]*/
        double CAR_DIM_X, CAR_DIM_Y, CAR_DIM_Z; /** car dimensions       [m]*/
        double CAR_BACK2AXLE;                   /** from back to axle    [m]*/
        double CAR_MASS;                        /** car weight           [kg]*/
        double CAR_FRONTAXLE_W;                 /** width of front axle  [m]*/
        double CAR_BACKAXLE_W;                  /** width of back axle   [m]*/
        double CAR_WHEEL_R;                     /** wheel radius         [m]*/

        double     speed    = 0;                /** speed                [ m/s   ]*/
        double     speedKMH = 0;                /** speed                [ Km/h  ]*/

        double     yawRate = 0;                 /** yaw rate             [ rad/s ]*/
        double     accelX  = 0;                 /** longitudinal acc     [ m/s2  ]*/
        double     accelY  = 0;                 /** lateral acc          [ m/s2  ]*/

        double     steerAngle     = 0;          /** steering wheel angle [ rad   ]*/
        double     steerAngleRate = 0;          /** steering wheel angle [ rad/s ]*/
        double     wheelAngle     = 0;          /** wheel angle          [ rad   ]*/

        uint8_t    brakePedalSts        = 0;    /** brake status         [ on/off ]*/
        double     brakeMasterPressure  = 0;    /** brake pressure       [ bar ]*/
        double     gasPedal             = 0;    /** gas padal percentage [ 0-1 ]*/
        double     engineTorque         = 0;    /** engine torque        [ % ]*/
        double     engineFrictionTorque = 0;    /** friction torque      [ % ]*/

        uint8_t    actualGear = 0;              /** current Gear         [ int ]*/
        uint16_t   RPM = 0;                     /** RPM                  [ int ]*/

        /**wheel speeds         [ m/s ]*/
        double     wheelFLspeed =0, wheelFRspeed =0, wheelRLspeed =0, wheelRRspeed =0;
        /**wheel dir            [int] ** 0:no_dir   1:forward   2:backward   4:invalid*/
        uint8_t    wheelFLDir =0, wheelFRDir =0, wheelRLDir =0, wheelRRDir =0;


        double     sideSlip     = 0;        /** beta angle               [ rad   ]*/
        uint8_t    tractionGrip = 0;        /** traction control grip    [ ? ]*/

        // faults
        uint8_t ESPFault = 0;

        // inputs
        uint8_t activateDrive = 0, activateSteer = 0;

        /** Odometry vales in space*/
        long double x = 0;
        long double y = 0;

        /** Direction of car in space*/
        long double carDirection = 0;

        /** Boolean variable for know if new data is arrived*/
        bool Bspeed = 0, Bsteer = 0;
        int  posBuffer = 0;

        #define ODOM_BUFFER_SIZE 10
        struct odom_t {
            long double x = 0;
            long double y = 0;
            long double z = 0;
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

        bool carOdometry(tk::common::Tfpose &tf, uint64_t time = 0){
            odom_t odom;
            bool status = carOdometry(odom, time);
            tf = tk::common::odom2tf(odom.x, odom.y, odom.yaw);
            return status;
        }

        void init() override{
            tk::data::SensorData::init();
            header.sensor = sensorName::VEHICLE;
        }

        void release() override {}

        bool checkDimension(SensorData *s) override {
            return true;
        }

        VehicleData& operator=(const VehicleData &s) {
            SensorData::operator=(s);

            this->CAR_WHEELBASE         = s.CAR_WHEELBASE;             
            this->CAR_DIM_X             = s.CAR_DIM_X;    
            this->CAR_DIM_Y             = s.CAR_DIM_Y;
            this->CAR_DIM_Z             = s.CAR_DIM_Z;
            this->CAR_BACK2AXLE         = s.CAR_BACK2AXLE;          
            this->CAR_MASS              = s.CAR_MASS;      
            this->CAR_FRONTAXLE_W       = s.CAR_FRONTAXLE_W;         
            this->CAR_BACKAXLE_W        = s.CAR_BACKAXLE_W;        
            this->CAR_WHEEL_R           = s.CAR_WHEEL_R;
            this->speed                 = s.speed;
            this->speedKMH              = s.speedKMH;
            this->yawRate               = s.yawRate;
            this->accelX                = s.accelX;
            this->accelY                = s.accelY;
            this->steerAngle            = s.steerAngle;
            this->steerAngleRate        = s.steerAngleRate;
            this->wheelAngle            = s.wheelAngle;
            this->brakePedalSts         = s.brakePedalSts;
            this->brakeMasterPressure   = s.brakeMasterPressure;
            this->gasPedal              = s.gasPedal;
            this->engineTorque          = s.engineTorque;
            this->engineFrictionTorque  = s.engineFrictionTorque;
            this->actualGear            = s.actualGear;
            this->RPM                   = s.RPM;
            this->wheelFLspeed          = s.wheelFLspeed;
            this->wheelFRspeed          = s.wheelFRspeed;
            this->wheelRLspeed          = s.wheelRLspeed;
            this->wheelRRspeed          = s.wheelRRspeed;
            this->wheelFLDir            = s.wheelFLDir;
            this->wheelFRDir            = s.wheelFRDir;
            this->wheelRLDir            = s.wheelRLDir;
            this->wheelRRDir            = s.wheelRRDir;
            this->sideSlip              = s.sideSlip;
            this->tractionGrip          = s.tractionGrip;
            this->ESPFault              = s.ESPFault;
            this->activateDrive         = s.activateDrive;
            this->activateSteer         = s.activateSteer;
            this->x                     = s.x;
            this->y                     = s.y;
            this->carDirection          = s.carDirection;
            this->Bspeed                = s.Bspeed;
            this->Bsteer                = s.Bsteer;
            this->posBuffer             = s.posBuffer;

            for(int i=0; i < ODOM_BUFFER_SIZE; i++){
                this->odometryBuffer[i].x   = s.odometryBuffer[i].x;
                this->odometryBuffer[i].y   = s.odometryBuffer[i].y;
                this->odometryBuffer[i].z   = s.odometryBuffer[i].z;
                this->odometryBuffer[i].yaw = s.odometryBuffer[i].yaw;
                this->odometryBuffer[i].t   = s.odometryBuffer[i].t;
            }

            return *this;
         }


        friend std::ostream& operator<<(std::ostream& os, const VehicleData& m) {
            os << m.header.stamp<< " speed: "<<m.speed<<" wheelangle: "<<m.wheelAngle;
            return os;
        }

  
        bool toVar(std::string name, tk::math::MatIO::var_t &var) {

            // save uint8 in matlab as double
            double actualGear_d = actualGear;
            double tractionGrip_d = tractionGrip;

            std::vector<tk::math::MatIO::var_t> structVars(25);
            structVars[ 0].set("stamp",              header.stamp);
            structVars[ 1].set("CAR_WHEELBASE",      CAR_WHEELBASE); 
            structVars[ 2].set("CAR_DIM_X",          CAR_DIM_X);
            structVars[ 3].set("CAR_DIM_Y",          CAR_DIM_Y);
            structVars[ 4].set("CAR_DIM_Z",          CAR_DIM_Z);
            structVars[ 5].set("CAR_BACK2AXLE",      CAR_BACK2AXLE);
            structVars[ 6].set("CAR_MASS",           CAR_MASS);
            structVars[ 7].set("CAR_FRONTAXLE_W",    CAR_FRONTAXLE_W);
            structVars[ 8].set("CAR_BACKAXLE_W",     CAR_BACKAXLE_W);
            structVars[ 9].set("CAR_WHEEL_R",        CAR_WHEEL_R);
            structVars[10].set("speed",              speed);
            structVars[11].set("yawRate",            yawRate);
            structVars[12].set("accelX",             accelX);
            structVars[13].set("accelY",             accelY);
            structVars[14].set("wheelAngle",         wheelAngle);
            structVars[15].set("brakeMasterPressure",brakeMasterPressure);
            structVars[16].set("gasPedal",           gasPedal);
            structVars[17].set("engineTorque",       engineTorque);
            structVars[18].set("actualGear_d",       actualGear_d);
            structVars[19].set("wheelFLspeed",       wheelFLspeed);
            structVars[20].set("wheelFRspeed",       wheelFRspeed);
            structVars[21].set("wheelRLspeed",       wheelRLspeed);
            structVars[22].set("wheelRLspeed",       wheelRLspeed);
            structVars[23].set("sideSlip",           sideSlip);
            structVars[25].set("tractionGrip_d",     tractionGrip_d);
            return var.setStruct(name, structVars);
        }

 
        bool fromVar(tk::math::MatIO::var_t &var) {
            if(var.empty())
                return false;

            // saved in matlab as double
            double actualGear_d;
            double tractionGrip_d;

            var["stamp"              ].get(header.stamp);
            var["CAR_WHEELBASE"      ].get(CAR_WHEELBASE); 
            var["CAR_DIM_X"          ].get(CAR_DIM_X);
            var["CAR_DIM_Y"          ].get(CAR_DIM_Y);
            var["CAR_DIM_Z"          ].get(CAR_DIM_Z);
            var["CAR_BACK2AXLE"      ].get(CAR_BACK2AXLE);
            var["CAR_MASS"           ].get(CAR_MASS);
            var["CAR_FRONTAXLE_W"    ].get(CAR_FRONTAXLE_W);
            var["CAR_BACKAXLE_W"     ].get(CAR_BACKAXLE_W);
            var["CAR_WHEEL_R"        ].get(CAR_WHEEL_R);
            var["speed"              ].get(speed);
            var["yawRate"            ].get(yawRate);
            var["accelX"             ].get(accelX);
            var["accelY"             ].get(accelY);
            var["wheelAngle"         ].get(wheelAngle);
            var["brakeMasterPressure"].get(brakeMasterPressure);
            var["gasPedal"           ].get(gasPedal);
            var["engineTorque"       ].get(engineTorque);
            var["actualGear_d"       ].get(actualGear_d);
            var["wheelFLspeed"       ].get(wheelFLspeed);
            var["wheelFRspeed"       ].get(wheelFRspeed);
            var["wheelRLspeed"       ].get(wheelRLspeed);
            var["wheelRLspeed"       ].get(wheelRLspeed);
            var["sideSlip"           ].get(sideSlip);
            var["tractionGrip_d"     ].get(tractionGrip_d);

            actualGear = actualGear_d;
            tractionGrip = tractionGrip_d;
            return true;
        }
    };
}
}