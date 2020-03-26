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
        
        std::vector<double> lateralLanesDeviation; //Lateral deviation of the vehicle with respect to each lane 
        double vehicleHeadingAngle; //Vehicle heading angle with respect to the road
        std::vector<double> laneCurvatures; //Lane curvature ahead of the vehicle

        //detection ranges ahead and behind the vehicle
        double lanesRangeDetection;
        double obstacleRangeDetection;

        void init(tk::perception::perceptionData perceptionData, double lanesRangeDetection, double obstaclesRangeDetection) 
            : perceptionData(perceptionData), lanesRangeDetection(lanesRangeDetection), obstacleRangeDetection(obstacleRangeDetection){
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