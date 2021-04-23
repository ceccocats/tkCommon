#include "tkCommon/communication/car/CarControl.h"
#include "tkCommon/rt/Task.h"

using namespace tk::communication;

bool CarControl::init(tk::communication::CanInterface *soc) {
    this->soc = soc;
    run = true;

    pidTorque.init(0,0,0,0,1);
    pidBrake.init(0,0,0,0,1);

    pthread_create(&writeTh, NULL, writeCaller, (void*)this);
    pthread_create(&readTh, NULL, readCaller, (void*)this);
    return true;
}

void CarControl::close() {
    run = false;
    pthread_join(writeTh, NULL);
    pthread_join(readTh, NULL);
}

void *CarControl::writeCaller(void *args) {
    ((CarControl*) args)->writeLoop();
}
void *CarControl::readCaller(void *args) {
    ((CarControl*) args)->readLoop();
}

void CarControl::computePIDs() {

    float velocity = odom.speed.x();
    torqueRequest = pidTorque.calculate(0.02,requestSpeed-velocity);
    brakeRequest  = pidBrake.calculate(0.02,velocity-requestSpeed);

    setBrakePos(brakeRequest*15000);
    setAccPos(torqueRequest*100);

    //maybe
    /*if(velocity < 1.0f){
        setBrakePos(0.6*15000);
    }*/
}

void CarControl::writeLoop() {
    tk::rt::Task t;
    t.init(20000);
    while(run) {

        if(usePid)
            computePIDs();

        /*if(velocity > -5) {
            pidTorque = pid.calculate(0.02,velocity-odom.speed.x());
            if(velocity<0)
                pidTorque -= 0.4;
            pidTorque = clamp<float>(pidTorque, -1, 1);

            if(pidTorque > 0) {
                setAccPos(pidTorque*100);
                setBrakePos((-0.5)*-15000);
            } else {    
                setAccPos(0);
                setBrakePos((-0.5+pidTorque)*-15000);
            }
        }*/
        requestSteerPos();
        requestAccPos();
        requestBrakePos();
        t.wait();
    }
}
void CarControl::readLoop() {

    tk::data::CanData data;
    while(run) {
        soc->read(&data);
    
        if(data.id() == GET_HW_ID+1) {
            uint8_t ecuId = data.frame.data[0];
            std::cout<<"ECU detected: "<<uint(ecuId)<<"\n";
        }
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
                odom.pose.x() = float(x)/2e4;
                odom.angle.z() = float(alfa)*1.54;
                odom.header.stamp = data.header.stamp;
            }
            if(data.id() == ODOM1_ID) {
                id = data.frame.data[1];
                memcpy(&y, &data.frame.data[1], 4);
                memcpy(&vel, &data.frame.data[1+4], 2);
                y =  __bswap_32(y);
                vel =  __bswap_16(vel);               
                odom.pose.y() = float(y)/2e4;
                odom.speed.x() = vel;
                odom.header.stamp = data.header.stamp;
            }
        }


    }
}


void CarControl::setSteerZero() {
    tk::data::CanData data;
    data.frame.can_dlc = 1;
    data.frame.can_id = SET_ZERO_STEERING_ID;
    data.frame.data[0] = steerECU;
    soc->write(&data);
    return;
}
void CarControl::resetSteerMotor() {
    tk::data::CanData data;
    data.frame.can_dlc = 0;
    data.frame.can_id = RESET_ERR_STEERING_ID;
    soc->write(&data);
    return;
}
void CarControl::sendGenericCmd(std::string cmd) {
    tkASSERT(cmd.size() < 8-2)

    tk::data::CanData data;
    data.frame.can_dlc = 8;
    data.frame.can_id = GENERIC_CMD_ID;
    
    data.frame.data[0] = steerECU;
    data.frame.data[1] = 0x80;
    for(int i=0; i<8-2; i++) {
        data.frame.data[i+2] = i < cmd.size() ? cmd[i] : ' ';
    }
    soc->write(&data);
}
void CarControl::setSteerPos(int32_t pos, uint8_t acc, uint16_t vel) {
    pos = __bswap_32(pos);
    vel = __bswap_16(vel);

    tk::data::CanData data;
    data.frame.can_dlc = 8;
    data.frame.can_id = SET_STEERING_ID;
    data.frame.data[0] = steerECU;
    data.frame.data[1] = acc;
    memcpy(&data.frame.data[2], &vel, 2);
    memcpy(&data.frame.data[4], &pos, 4);
    soc->write(&data);
    return;
}
void CarControl::setAccPos(uint16_t pos) {
    pos = __bswap_16(pos);

    tk::data::CanData data;
    data.frame.can_dlc = 3;
    data.frame.can_id = SET_ACC_ID;
    data.frame.data[0] = accECU;
    memcpy(&data.frame.data[1], &pos, 2);
    soc->write(&data);
    return;
}
void CarControl::setBrakePos(uint16_t pos) {
    pos = __bswap_16(pos);

    tk::data::CanData data;
    data.frame.can_dlc = 3;
    data.frame.can_id = SET_BRAKE_ID;
    data.frame.data[0] = brakeECU;
    memcpy(&data.frame.data[1], &pos, 2);
    soc->write(&data);
    return;
}
void CarControl::sendAccEnable(bool status) {
    tk::data::CanData data;
    data.frame.can_dlc = 2;
    data.frame.can_id = ACC_RELE_ID;
    data.frame.data[0] = accECU;
    data.frame.data[1] = status;
    soc->write(&data);
}

void CarControl::sendOdomEnable(bool status) {
    tk::data::CanData data;
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

void CarControl::sendEngineStart() {
    tk::data::CanData data;
    data.frame.can_dlc = 1;
    data.frame.can_id = ON_OFF_ID;
    data.frame.data[0] = steerECU;
    soc->write(&data);
}

void CarControl::requestSteerPos() {
    tk::data::CanData data;
    data.frame.can_dlc = 1;
    data.frame.can_id = GET_STEERING_ID;
    data.frame.data[0] = steerECU;
    soc->write(&data); 
}

void CarControl::requestAccPos() {
    tk::data::CanData data;
    data.frame.can_dlc = 1;
    data.frame.can_id = GET_ACC_ID;
    data.frame.data[0] = accECU;
    soc->write(&data);
}

void CarControl::requestBrakePos() {
    tk::data::CanData data;
    data.frame.can_dlc = 1;
    data.frame.can_id = GET_BRAKE_ID;
    data.frame.data[0] = brakeECU;
    soc->write(&data);
}

void CarControl::requestMotorId() {
    tk::data::CanData data;
    data.frame.can_dlc = 0;
    data.frame.can_id = GET_HW_ID;
    soc->write(&data);
}

void CarControl::setVel(float vel){
    requestSpeed = vel;
}