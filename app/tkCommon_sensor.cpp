#include "tkCommon/sensor/Sensor.h"
#include "tkCommon/data/ImuData.h"
#include "tkCommon/data/GpsData.h"

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
            if (senStatus == SensorStatus::ONLINE) {
                // DO stuff
            } else {
                // DO stuff
            }

            info.type = tk::data::DataType::GPSIMU;
            
            addPool<tk::data::GpsData>(0);
            addPool<tk::data::GpsData>(1);
            addPool<tk::data::ImuData>();

            tkDBG("Init succesful.\n");
            return true;
        }
        bool closeChild()
        {
            tkDBG("Closed.\n");
            return true;
        }
        bool readOnline(tk::data::SensorData* data)
        {   
            if (data->header.type == tk::data::DataType::GPS) {
                auto gps = dynamic_cast<tk::data::GpsData*>(data);
                //tkDBG("Scrivo un gps.\n");
                sleep(1);
                return true;
            }
            
            if (data->header.type == tk::data::DataType::IMU) {
                auto imu = dynamic_cast<tk::data::ImuData*>(data);
                //tkDBG("Scrivo un imu.\n");
                usleep(100000);
                return true;
            }

            return false;
        }
        bool readLog(tk::data::SensorData* data)
        {
            return true;
        }
    };
}}

//-----------------------------------------------------------------------------

tk::sensors::MySensor   sensor;
YAML::Node              conf;
bool gRun = true;

void gpsGrab(uint32_t sensorID) {
    //sleep(2);
    int idx;
    const tk::data::GpsData *data;
    while(gRun) {
        if (sensor.grab<tk::data::GpsData>(data, idx, 10000000, sensorID)) {
            tkWRN("Leggo GPS "<<(int) sensorID<<"\n");
            sensor.release<tk::data::GpsData>(idx);
        } else {
            tkERR("aspetto\n");
            usleep(10000);
        }
    }
}

void imuGrab() {
    //sleep(2);
    int idx;
    const tk::data::ImuData *data;
    while(gRun) {
        if (sensor.grab<tk::data::ImuData>(data, idx, 1000000)) {
            tkWRN("Leggo IMU\n");
            sensor.release<tk::data::ImuData>(idx);
        } else {
            tkERR("aspetto\n");
            usleep(10000);
        }
    }
}

void signal_handler(int signal)
{
    tkMSG("\nRequest closing..\n");
    gRun = false;
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    

    // init and start sensor
    if (!sensor.init(conf, "test", nullptr)) {
        return 0;
    }
    sensor.start();

    std::thread imuTH(imuGrab);
    std::thread gps0TH(gpsGrab, 0);
    std::thread gps1TH(gpsGrab, 1);
    
    //---------------------------------------------

    imuTH.join();
    gps0TH.join();
    gps1TH.join();

    sensor.close();
    return 0;
}