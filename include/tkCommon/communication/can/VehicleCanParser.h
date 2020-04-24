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
                uint64_t x = *frame.data();

                // decode
                x = reverse_byte_order(x);
                uint64_t lmask = 0xffffffffffffffff >> 64 - sig.getLength();
                int      shift = 64 - (7 - sig.getStartbit()%8 + 8*(sig.getStartbit()/8) + sig.getLength());
                x = (x >> shift) & lmask;

                double val = double(x)*sig.getFactor() + sig.getOffset();
            
                //std::cout<<sig.getName()<<": "<<val<<"\n";
            }

        }
    };
}}