#pragma once
#include "tkCommon/data/gen/KistlerData_gen.h"

namespace tk { namespace data {
    enum class KistlerMsgType : uint8_t {
        VelX_VelY_Vel_Angle = 0,
        Distance = 1,
        Pitch_Roll = 2, 
        AccHor_AccCBody = 3,
        AngVelHor = 4,
        Correvit = 5,
        AccBody = 6,
        AngVelBody = 7,
        Empty = 255
    };

    class KistlerData : public KistlerData_gen {
    public:
         KistlerData() = default;
        ~KistlerData() = default;
        
        static KistlerMsgType fromCanID(const unsigned int aID) {
            switch (aID) {
            case 0x9fffffe0: return KistlerMsgType::VelX_VelY_Vel_Angle;
            case 0x9fffffe1: return KistlerMsgType::Distance;
            case 0x9fffffe2: return KistlerMsgType::Pitch_Roll;
            case 0x9fffffe3: return KistlerMsgType::AccHor_AccCBody;
            case 0x9fffffe4: return KistlerMsgType::AngVelHor;
            case 0x9fffffe5: return KistlerMsgType::Correvit;
            case 0x9fffffe6: return KistlerMsgType::AccBody;
            case 0x9fffffe7: return KistlerMsgType::AngVelBody;
            //case 0x9fffffe8: return KistlerData_gen::;
            default: return KistlerMsgType::Empty; 
            }
        }
    };
}}