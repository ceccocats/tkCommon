#pragma once
#include "tkCommon/communication/CanInterface.h"

namespace tk { namespace communication {

    class CarControl {
    private:
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

        tk::communication::CanInterface *soc = nullptr;
        pthread_t writeTh, readTh;
        bool run = true;

        // ecu ids
        uint8_t steerECU = 15;
        uint8_t accECU = 6;

    public:
        // car odometry
        tk::data::VehicleData::odom_t odom;
        int steerPos = 0;
        int brakePos = 0;
        int accPos = 0;

        bool init(tk::communication::CanInterface *soc) {
            this->soc = soc;
            run = true;
            pthread_create(&writeTh, NULL, writeCaller, (void*)this);
            pthread_create(&readTh, NULL, readCaller, (void*)this);
            return true;
        }
        void close() {
            run = false;
            pthread_join(writeTh, NULL);
            pthread_join(readTh, NULL);
        }

        static void *writeCaller(void *args) {
            ((CarControl*) args)->writeLoop();
        }
        static void *readCaller(void *args) {
            ((CarControl*) args)->readLoop();
        }

        void writeLoop() {
            LoopRate rate(20000);
            while(run) {
                requestSteerPos();
                requestAccPos();
                requestBrakePos();
                rate.wait();
            }
        }
        void readLoop() {

            tk::data::CanData_t data;
            while(run) {
                soc->read(&data);
            
                if(data.id() == GET_STEERING_ID+1) {
                    int32_t pos;
                    memcpy(&pos, &data.frame.data[1], 4);
                    steerPos = __bswap_32(pos); 
                }
                if(data.id() == GET_ACC_ID+1) {
                    uint16_t pos;
                    memcpy(&pos, &data.frame.data[1], 2);
                    accPos = __bswap_16(pos); 
                }
                if(data.id() == GET_BRAKE_ID+1) {
                    uint16_t pos;
                    memcpy(&pos, &data.frame.data[1], 2);
                    brakePos = __bswap_16(pos); 
                }
                if(data.id() == ODOM0_ID || data.id() == ODOM1_ID) {
                    uint8_t id;
                    int32_t x, y;
                    uint16_t alfa, vel;
                    if(data.id() == ODOM0_ID) {
                        id = data.frame.data[0];
                        memcpy(&x, &data.frame.data[1], 4);
                        memcpy(&alfa, &data.frame.data[1+4], 2);
                        x =  __bswap_32(x);
                        alfa =  __bswap_16(alfa);               
                        odom.x = float(x)/2e4;
                        odom.yaw = float(alfa)*1.54;
                        odom.t = data.stamp;
                    }
                    if(data.id() == ODOM1_ID) {
                        id = data.frame.data[1];
                        memcpy(&y, &data.frame.data[1], 4);
                        memcpy(&vel, &data.frame.data[1+4], 2);
                        y =  __bswap_32(y);
                        vel =  __bswap_16(vel);               
                        odom.y = float(y)/2e4;
                        odom.t = data.stamp;
                    }
                }

        
            }
        }


        void setSteerZero() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 1;
            data.frame.can_id = SET_ZERO_STEERING_ID;
            data.frame.data[0] = steerECU;
            soc->write(&data);
            return;
        }
        void resetSteerMotor() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 0;
            data.frame.can_id = RESET_ERR_STEERING_ID;
            soc->write(&data);
            return;
        }
        void sendGenericCmd(std::string cmd) {
            tkASSERT(cmd.size() < 8-2)

            tk::data::CanData_t data;
            data.frame.can_dlc = 8;
            data.frame.can_id = GENERIC_CMD_ID;
            
            data.frame.data[0] = steerECU;
            data.frame.data[1] = 0x80;
            for(int i=0; i<8-2; i++) {
                data.frame.data[i+2] = i < cmd.size() ? cmd[i] : ' ';
            }
            soc->write(&data);
        }
        void setSteerPos(int32_t pos, uint8_t acc = 0, uint16_t vel = 0) {
            pos = __bswap_32(pos);
            vel = __bswap_16(vel);

            tk::data::CanData_t data;
            data.frame.can_dlc = 8;
            data.frame.can_id = SET_STEERING_ID;
            data.frame.data[0] = steerECU;
            data.frame.data[1] = acc;
            memcpy(&data.frame.data[2], &vel, 2);
            memcpy(&data.frame.data[4], &pos, 4);
            soc->write(&data);
            return;
        }
        void setAccPos(uint16_t pos) {
            pos = __bswap_16(pos);

            tk::data::CanData_t data;
            data.frame.can_dlc = 3;
            data.frame.can_id = SET_ACC_ID;
            data.frame.data[0] = accECU;
            memcpy(&data.frame.data[1], &pos, 2);
            soc->write(&data);
            return;
        }
        void setBrakePos(uint16_t pos) {
            tk::data::CanData_t data;
            data.frame.can_dlc = 3;
            data.frame.can_id = SET_BRAKE_ID;
            data.frame.data[0] = steerECU;
            memcpy(&data.frame.data[1], &pos, 2);
            soc->write(&data);
            return;
        }
        void sendAccEnable(bool status) {
            tk::data::CanData_t data;
            data.frame.can_dlc = 2;
            data.frame.can_id = ACC_RELE_ID;
            data.frame.data[0] = accECU;
            data.frame.data[1] = status;
            soc->write(&data);
        }

        void sendOdomEnable(bool status) {
            tk::data::CanData_t data;
            data.frame.can_dlc = 2;
            data.frame.can_id = SET_ACTIVE_ID;
            data.frame.data[0] = accECU;
            data.frame.data[1] = status;
            soc->write(&data);

            // reset 
            data.frame.can_dlc = 0;
            data.frame.can_id = ODOM_RESET_ID;
            soc->write(&data);
        }


    private:
        void requestSteerPos() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 1;
            data.frame.can_id = GET_STEERING_ID;
            data.frame.data[0] = steerECU;
            soc->write(&data); 
        }

        void requestAccPos() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 1;
            data.frame.can_id = GET_ACC_ID;
            data.frame.data[0] = accECU;
            soc->write(&data);
        }

        void requestBrakePos() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 1;
            data.frame.can_id = GET_BRAKE_ID;
            data.frame.data[0] = accECU;
            soc->write(&data);
        }

        /*
        uint8_t getMotorId() {
            tk::data::CanData_t data;
            data.frame.can_dlc = 0;
            data.frame.can_id = GET_HW_ID;
            soc->write(&data);
            soc->read(&data);
            tkASSERT(data.id() == GET_HW_ID+1);
            return data.frame.data[0];
        }
        */
    };
    
}}