#pragma once

#include "tkCommon/sensor/Sensor.h"

namespace tk { namespace sensors {
    class SensorSpawner 
    {
    public:
        virtual bool spawn(const std::string &list, std::map<std::string,tk::sensors::Sensor*> *sensors) = 0;
    private:
    };
}}
