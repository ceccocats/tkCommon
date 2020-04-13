#pragma once
#include "tkCommon/common.h"
#include "tkCommon/math/CSpline2D.h"
#include "tkCommon/data/ActuationData.h"

namespace tk { namespace data {

/**
 * Controller Output Data
 */
class ControlOutputData{
    public:
        ActuationData actuation;
        tk::math::CSpline2D trajectory;

        ControlOutputData();
        
        void init(ActuationData actuation, tk::math::CSpline2D trajectory) 
        : actuation(actuation)
        , trajectory(trajectory){};
        
};

}
}