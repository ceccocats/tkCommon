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
        tk::perception::perceptionData perceptionData; //boxes, signals, lanes and vehicle data
        //VehicleData vehicleData;
        //vector<generic> obstacles;
        std::vector<double> lateralLanesDeviation; //Lateral deviation of the vehicle with respect to each lane 
        double vehicleHeadingAngle; //Vehicle heading angle with respect to the road
        std::vector<double> laneCurvatures; //Lane curvature ahead of the vehicle

        void init(tk::perception::perceptionData perceptionData) 
            : perceptionData(perceptionData){
                calcLateralLanesDeviation();
                calcVehicleHeadingAngle();
                calcLaneCurvatures();
            };

    private:
        void calcLateralLanesDeviation(){
            //TO-DO
        };

        void calcVehicleHeadingAngle(){
            //TO-DO
        };

        void calcLaneCurvatures(){
            //TO-DO
        };

};


}
}