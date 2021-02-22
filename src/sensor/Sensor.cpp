#include "tkCommon/sensor/Sensor.h"

namespace tk { namespace sensors {
Clock::Clock()
{
    initted = false;
}

Clock::~Clock()
{

}
        
void 
Clock::init(const YAML::Node conf)
{   
    if (!initted) {
        std::string port    = tk::common::YAMLgetConf<std::string>(conf, "port", "/dev/ttyUSB0");
        int         baud    = tk::common::YAMLgetConf<int>(conf, "baud", 9600);
        
        // open serial port
        if (!serial.init(port, baud)) {
            tkWRN("Cannot open communication with SynchBox.\n");
            initted = false;
        } else {
            initted = true;
        }
    }
}

void 
Clock::start()
{

}

void 
Clock::stop()
{

}

timeStamp_t 
Clock::getSychTimeStamp(int frameCounter, int triggerLine)
{
    tkASSERT(initted == true);
    tkASSERT(triggerLine >= 0 && triggerLine < freq.size(), "Out of bounds.");

    return (timeStamp_t) t0 + (frameCounter * 1.0f/freq[triggerLine]); 
}



void 
Sensor::start() 
{
    // start loop thread
    pthread_attr_init(&attr);
    pthread_create(&t0, &attr, &Sensor::loop_helper, this);
    readLoopStarted = true;
}

void* 
Sensor::loop_helper(void *context) 
{
    return ((tk::sensors::Sensor*)context)->loop(nullptr);
}

void* 
Sensor::loop(void *vargp) 
{
    while (senStatus != SensorStatus::STOPPING && senStatus != SensorStatus::ERROR) {
        if (!readFrame()) {
            tkWRN("Error while reading. Trying again in 2 seconds...\n");
            sleep(2);
            continue;
        }
    }
    pthread_exit(nullptr);
}

tk::common::Tfpose 
Sensor::getTf(int id) 
{
    if(id >= tf.size())
        return tk::common::Tfpose::Identity();
    else
        return tf[id];
}

std::vector<tk::common::Tfpose> 
Sensor::getTfs() 
{
    return tf;
}

bool 
tk::sensors::Sensor::close() 
{
    // stop recording
    if (senStatus == SensorStatus::RECORDING)
        stopRecord();

    // stop online reading thread
    senStatus = SensorStatus::STOPPING;
    if(readLoopStarted)
        pthread_join(t0, nullptr);
    
    // stop log
	if (senStatus == SensorStatus::OFFLINE)
		log->stop();
    
    // clear pool
    pool.close();

    return true;
}

bool 
Sensor::readFrame() 
{
    // get data from pool
    bool    retval;
    int     idx;
    tk::data::SensorData* data = pool.add(idx);

    // read data
    if (senStatus != SensorStatus::OFFLINE) {
        retval = readOnline(data);
    } else {         
        retval = readLog(data);
        if(retval)
            log->wait(data->header.stamp, info.synched);
        else
            info.synched = true;
    }


    if (poolEmpty && retval)
        poolEmpty = false;

    if (retval)   
        info.dataArrived++;

    // fill data header
    if(data->header.name != info.name)
        data->header.name = info.name;
    data->header.tf         = getTf();
    data->header.messageID  = info.dataArrived;

    // release pool
    pool.releaseAdd(idx);
    
    return retval;
}

tk::sensors::SensorStatus 
Sensor::status()
{
    return this->senStatus;
}

void
Sensor::release(const int idx) 
{
    pool.releaseGet(idx);
}
}}