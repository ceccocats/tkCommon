#pragma once
#include "tkCommon/data/gen/VehicleData_gen.h"

namespace tk { namespace data {

    class VehicleData : public VehicleData_gen {

        public:
            VehicleData() {}
            ~VehicleData() {}

            // odom compute vars
            double x = 0, y = 0, carDirection = 0;
            bool Bspeed = false, Bsteer = false;
    };
}
}