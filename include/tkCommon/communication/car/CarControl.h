#pragma once
#include "tkCommon/communication/CanInterface.h"
#include "tkCommon/data/OdomData.h"
#include "tkCommon/PID.h"

namespace tk { namespace communication {

    class CarControl {
    private:
        // can messages IDs
        static const canid_t GET_HW_ID             = ( 0x7fe | CAN_EFF_FLAG );
        static const canid_t SET_ZERO_STEERING_ID  = ( 0x202 | CAN_EFF_FLAG );
        static const canid_t SET_STEERING_ID       = ( 0x204 | CAN_EFF_FLAG );
        static const canid_t GET_STEERING_ID       = ( 0x206 | CAN_EFF_FLAG );
        static const canid_t RESET_ERR_STEERING_ID = ( 0x20c | CAN_EFF_FLAG );
        static const canid_t SET_BRAKE_ID          = ( 0x208 | CAN_EFF_FLAG );
        static const canid_t GET_BRAKE_ID          = ( 0x20a | CAN_EFF_FLAG );
        static const canid_t SET_ACC_ID            = ( 0x102 | CAN_EFF_FLAG );
        static const canid_t GET_ACC_ID            = ( 0x104 | CAN_EFF_FLAG );
        static const canid_t GENERIC_CMD_ID        = ( 0x20e | CAN_EFF_FLAG );
        static const canid_t ACC_RELE_ID           = ( 0x11c | CAN_EFF_FLAG );
        static const canid_t SET_ACTIVE_ID         = ( 0x110 | CAN_EFF_FLAG );
        static const canid_t ODOM_RESET_ID         = ( 0x11a | CAN_EFF_FLAG );
        static const canid_t ODOM0_ID              = ( 0x121 | CAN_EFF_FLAG );
        static const canid_t ODOM1_ID              = ( 0x122 | CAN_EFF_FLAG );
        static const canid_t ON_OFF_ID             = ( 0x210 | CAN_EFF_FLAG );

        // can iface
        tk::communication::CanInterface *soc = nullptr;
        
        pthread_t writeTh, readTh;
        bool run = true;

        // ecu ids
        uint8_t brakeECU = 3;
        uint8_t steerECU = 16;
        uint8_t accECU   = 6;

        bool active = false;

        float targetBrake = 0;      /**< requested target val */ 
        float targetThrottle = 0;   /**< requested target val */ 
        float targetSteer = 0;      /**< requested target val */     
        float actBrake = 0;         /**< currently actuation val */ 
        float actThrottle = 0;      /**< currently actuation val */ 
        float actSteer = 0;         /**< currently actuation val */     

        int steerOffset = 0;   /** STEER position OFFSET distance from auto ZERO */
        uint8_t steerAcc = 0;  /** ACC   param steer motor */
        uint16_t steerVel = 0; /** SPEED param steer motor */  

        static void *writeCaller(void *args);
        static void *readCaller(void *args);

        void writeLoop();
        void readLoop();

    public:
        tk::data::OdomData odom; /**< current vehicle odometry */
        int steerPos = 0;                   /**< current steer value read by ECU */
        int brakePos = 0;                   /**< current brake value read by ECU */
        int accPos = 0;                     /**< current accel value read by ECU */


        /**
         *  Init the sistem on a specified CAN socket
         *  it start 2 thread to read and write can messages
         **/
        bool init(tk::communication::CanInterface *soc);
        /**
         *  Stop and close the system
         */ 
        void close();

        /**
         * Set the current position of the steer as Zero
         */
        void setSteerZero();
        /**
         * Reser steer motor errors
         */ 
        void resetSteerMotor();
        /**
         *  Send a command to the steer motor
         *  ex: "OFF" will torn off the motor, ask SpinItalia if you need other commands
         */
        void sendGenericCmd(std::string cmd);

        /**
         *  Enable actuation
         */
        void enable(bool status);
        
        /**
         *  Set Odometry stream enabled or not
         */
        void sendOdomEnable(bool status);
        /**
         *  Simulate pression off engine start
         */
        void sendEngineStart();

        /**
         * Send request to send ID to each ECUs
         */ 
        void requestMotorId();

        /**
         * set steer parameters
         */
        void setSteerParams(int offset, int acc, int vel)  {
            steerOffset = clamp<int>(offset, -2000, 2000);
            steerAcc = clamp<int>(acc, 0, 255); 
            steerVel = clamp<int>(vel, 0, 1000); 
        }
        int getSteerOffset()    { return steerOffset; }
        uint8_t getSteerAcc()   { return steerAcc;    }
        uint16_t getSteerVel()  { return steerVel;    }

        /** target steer angle DEG */
        void setTargetSteer(float val)    { targetSteer = clamp<float>(val, -30, 30); }
        /** target Brake 0-1 */
        void setTargetBrake(float val)    { targetBrake = clamp<float>(val, 0, 1); }
        /** target Throttle 0-1 */
        void setTargetThrottle(float val) { targetThrottle = clamp<float>(val, 0, 1); }

        /** get last actuation */
        float getActSteer() { return actSteer; }
        /** get last actuation */
        float getActThrottle() { return actThrottle; }
        /** get last actuation */
        float getActBrake() { return actBrake; }

private:   
        std::mutex cmd_lock;

        // request info from the system
        void requestSteerPos();
        void requestAccPos();
        void requestBrakePos();
        void computePIDs();

        /**
         *  Send a steer position command to the motor
         *  from -22000 to +22000 on maserati (it depends on the steering weel)
         *  acc: 0 - 255
         *  vel: 0 - 1000
         *  !!! WARNING for performace issue ACC and VEL should be setted only once, passing 0 keep the setting
         */
        void setSteerPos(int32_t pos, uint8_t acc = 0, uint16_t vel = 0);

        /**
         *  Send a gas position command to the motor
         *  0 - 1 
         */
        void setAccPos(float val);
        /**
         *  Send a brake position command to the motor
         *  0 - 1
         */
        void setBrakePos(float val);

        /** 
         * Set steer angle (deg) (Negative: Sx)
         */
        void setSteerAngle(float angle);
    };
    
}}
