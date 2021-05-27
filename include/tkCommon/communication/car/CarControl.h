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

    public:
        tk::data::OdomData odom; /**< current vehicle odometry */
        int steerPos = 0;                   /**< current steer value      */
        int brakePos = 0;                   /**< current brake value      */
        int accPos = 0;                     /**< current accel value      */

        tk::common::PID pidTorque;
        tk::common::PID pidBrake;
        float torqueRequest, brakeRequest;

        bool usePid = false;

        float requestSpeed = 0;

        /**
         *  Init the sistem on a specified CAN socket
         *  it start 2 thread to read and write can messages
         **/
        bool init(tk::communication::CanInterface *soc);
        /**
         *  Stop and close the system
         */ 
        void close();

        static void *writeCaller(void *args);
        static void *readCaller(void *args);

        void writeLoop();
        void readLoop();

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
         *  Send a steer position command to the motor
         *  from -18000 to +18000 on maserati (it deoends on the steering weel)
         */
        void setSteerPos(int32_t pos, uint8_t acc = 0, uint16_t vel = 0);
        /**
         *  Send a gas position command to the motor
         *  0 is no gas, 100 is full gas
         */
        void setAccPos(uint16_t pos);
        /**
         *  Send a brake position command to the motor
         *  0 is no brake, 20000 is full brake
         *  4000 step for 6 mm
         *  max 20000 step for 30 mm  
         */
        void setBrakePos(uint16_t pos);
        /**
         *  Set Accel (and Brake?) enabled or not
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
         * Set vel
         * 
         */
        void setVel(float vel);

        void steerAngle(float angle, uint16_t vel = 0){
            float diff = -angle/0.0015;
            setSteerPos(diff,0,vel);
        }

        void requestMotorId();

private:        
        // request info from the system
        void requestSteerPos();
        void requestAccPos();
        void requestBrakePos();
        void computePIDs();
    };
    
}}
