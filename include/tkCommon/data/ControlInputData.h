#pragma once
#include "tkCommon/common.h"
#include "tkCommon/data/VehicleData.h"
#include "tkCommon/data/PerceptionData.h"
#include <vector>

namespace tk { namespace data {

/**
 * Controller Input Data
 */
class ControlInputData{
    public:
        tk::perception::perceptionData perceptionData; //obstacles, road signals, lanes and vehicle data

        void init(tk::perception::perceptionData perceptionData) : perceptionData(perceptionData){};

};


}
}