#include "tkCommon/sensor/Sensor.h"

namespace tk { namespace sensors {
    class MySensor : public Sensor {
    public:
        MySensor()
        {

        }
        ~MySensor()
        {

        }
        bool startRecord(const std::string &fileName)
        {
            return true;
        }
        bool stopRecord()
        {
            return true;
        }
    private:
        bool initChild(const YAML::Node conf, const std::string &name, LogManager *log = nullptr)
        {
            return true;
        }
        bool closeChild()
        {
            return true;
        }
        bool readOnline(tk::data::SensorData* data)
        {
            return true;
        }
        bool readLog(tk::data::SensorData* data)
        {
            return true;
        }
    };
}}

bool gRun = true;

int main(int argc, char** argv) {
    tk::sensors::MySensor sensor;
    return 0;
}