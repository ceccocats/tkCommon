#pragma once

#include <tkCommon/communication/can/dbciterator.hpp>
#include "tkCommon/data/CanData.h"
#include "tkCommon/data/VehicleData.h"

namespace tk { namespace communication {
    class VehicleCanParser {
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
            if(frame.id() != 326)
                return;

            for(auto sig : msgs[frame.id()]) {
                std::cout<<sig.getName()<<"\n";
                uint64_t x = *frame.data();
                
                double val = x;
                std::cout<<x<<"\n";
            }
        }
    };
}}