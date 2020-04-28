#pragma once

#include <tkCommon/communication/can/dbciterator.hpp>
#include "tkCommon/data/CanData.h"
#include "tkCommon/data/VehicleData.h"

namespace tk { namespace communication {

    static inline uint64_t reverse_byte_order(uint64_t x)
    {
        x = (x & 0x00000000FFFFFFFF) << 32 | (x & 0xFFFFFFFF00000000) >> 32;
        x = (x & 0x0000FFFF0000FFFF) << 16 | (x & 0xFFFF0000FFFF0000) >> 16;
        x = (x & 0x00FF00FF00FF00FF) << 8  | (x & 0xFF00FF00FF00FF00) >> 8;
        return x;
    }
    static inline void vehicleCalculateOdometry(tk::data::VehicleData *veh, uint64_t t, float freq = 0.02){
        //data read
        //actual angle of immaginary central wheel
        double wheelsAngle = veh->wheelAngle;
        //distance from car velocity
        double dist = (veh->speed)*freq;
        //radius of steering circle
        double radius = veh->CAR_WHEELBASE/sin(wheelsAngle);
        radius=fabs(radius);

        //circumference
        double circumference = radius * 2 * M_PI;
        if(dist != 0){
            if(wheelsAngle > -0.001 && wheelsAngle < 0.001){
                veh->x += sin(veh->carDirection)*dist;
                veh->y += cos(veh->carDirection)*dist;
            }else{
                //angle of car rotation
                double angle = (dist / circumference)  * 2 * M_PI;
                //distance from last and new point
                double di = sin(angle/2)*2*radius;
                //tempo of last carDirection
                double temp = veh->carDirection;

                if(wheelsAngle > 0){
                    //destra
                    veh->carDirection-=angle;
                }else{
                    //sinistra
                    veh->carDirection+=angle;
                }

                //direction of car
                double dir = (temp+veh->carDirection)/2;
                veh->x += sin(dir)*di;
                veh->y += cos(dir)*di;
            }
        }

        //write data
        veh->odom.x = veh->x;
        veh->odom.y = veh->y;
        veh->odom.yaw = -veh->carDirection +M_PI/2;
        veh->odom.t = t;
    }


    class VehicleCanParser {
    private:
        tk::data::VehicleData tmpVeh;

    public:
        std::map<int, Message> msgs;

         VehicleCanParser() {}
        ~VehicleCanParser() {}

        bool init(std::string dbc_file) {
            msgs.clear();

            try {
                DBCIterator dbc(dbc_file);
                for(auto message : dbc) {
                    msgs[message.getId()] = message;
                }
            } catch (const std::exception & ex) {
                std::cout << "Error reading DBC\n";
                std::cout << ex.what() << std::endl;
            }

            std::cout<<"Number of msgs: "<<msgs.size()<<"\n";
            for(auto m : msgs) {
                printMsg(m.first);
            }
        }

        void printMsg(int id) {
            std::cout<<"Msg "<<id<<"   "<<msgs[id].getName()<<"\n";
            for(auto sig : msgs[id]) {
                std::cout<<"\t"<<sig.getName()<<" [ "<<sig.getUnit()<<" ] "
                <<"\t\tval = val * "<<sig.getFactor()<<" + "<< sig.getOffset()<<"\n";
            }
            std::cout<<"\n";
        }

        /**
         * @brief Parse can frame and write dato to Vehicle
         * @param frame input can frame
         * @param vehData output vehicle data
         */
        void parse(tk::data::CanData_t &frame, tk::data::VehicleData &vehData) {
            
            if(msgs.count(frame.id()) == 0)
                return;

            for(auto sig : msgs[frame.id()]) {
                uint64_t x = *frame.data();

                // check dlc
                tkASSERT(frame.frame.can_dlc == msgs[frame.id()].getDlc());
                tkASSERT(sig.getByteOrder() == tk::communication::ByteOrder::MOTOROLA);

                // decode
                int sigLength = sig.getLength();
                x = reverse_byte_order(x);
                uint64_t lmask = 0xffffffffffffffff >> 64 - sigLength;
                int      shift = 64 - (7 - sig.getStartbit()%8 + 8*(sig.getStartbit()/8) + sigLength);
                x = (x >> shift) & lmask;

                double val;
                if(sig.getSign() == tk::communication::Sign::UNSIGNED) {
                    val = x;
                } else {
                    switch(sigLength) {
                    case  8: val =  int8_t(x); break;
                    case 16: val = int16_t(x); break;
                    case 32: val = int32_t(x); break;
                    case 64: val = int64_t(x); break;
                    default:                         
                        tkFATAL("SIGNED integer not supported with lenght: " + std::to_string(sigLength) + "\n");
                        break;
                    }
                }
                val = val*sig.getFactor() + sig.getOffset();

                //std::cout<<sig.getName()<<": "<<val<<"\n";

                // TODO: faster impl
                if(sig.getName() == "speedKMH") { 
                    vehData.speedKMH = val; 
                    vehData.speed = val/3.6;
                    vehData.Bspeed = true;
                    
                    //odometry calculation
                    if(vehData.Bspeed && vehData.Bsteer){
                        vehicleCalculateOdometry(&vehData, frame.stamp);
                        vehData.Bspeed = false;
                        vehData.Bsteer = false;
                    }
                } else if(sig.getName() == "wheelAngle")  {
                    vehData.wheelAngle = val/180.0*M_PI;
                    vehData.Bsteer = true;
                }      
                else if(sig.getName() == "yawRate")             vehData.yawRate = val/180.0*M_PI;
                else if(sig.getName() == "accelX")              vehData.accelX = val;
                else if(sig.getName() == "accelY")              vehData.accelY = val;
                else if(sig.getName() == "steerAngle")          vehData.steerAngle = val/180.0*M_PI;
                else if(sig.getName() == "steerAngleRate")      vehData.steerAngleRate = val/180.0*M_PI;
                else if(sig.getName() == "brakePedalSts")       vehData.brakePedalSts = val;
                else if(sig.getName() == "brakeMasterPressure") vehData.brakeMasterPressure = val;
                else if(sig.getName() == "gasPedal")            vehData.gasPedal = val;
                else if(sig.getName() == "engineTorque")        vehData.engineTorque = val;
                else if(sig.getName() == "actualGear")          vehData.actualGear = val;
                else if(sig.getName() == "RPM")                 vehData.RPM = val;
                else if(sig.getName() == "wheelFLspeed")        vehData.wheelFLspeed = val/3.6;
                else if(sig.getName() == "wheelFRspeed")        vehData.wheelFRspeed = val/3.6;
                else if(sig.getName() == "wheelRLspeed")        vehData.wheelRLspeed = val/3.6;
                else if(sig.getName() == "wheelRRspeed")        vehData.wheelRRspeed = val/3.6;
                else if(sig.getName() == "wheelFLDir")          vehData.wheelFLDir = val;
                else if(sig.getName() == "wheelFRDir")          vehData.wheelFRDir = val;
                else if(sig.getName() == "wheelRLDir")          vehData.wheelRLDir = val;
                else if(sig.getName() == "wheelRRDir")          vehData.wheelRRDir = val;   
                else if(sig.getName() == "sideSlip")            vehData.sideSlip = val;   
                else if(sig.getName() == "tractionGrip")        vehData.tractionGrip = val;   
            }

        }
    };
}}