#include "tkCommon/sensor/Sensor.h"

namespace tk { namespace sensors {
Clock::Clock()
{
    synched = false;
}

Clock::~Clock()
{

}
        
void 
Clock::init(const YAML::Node conf)
{   
    port    = tk::common::YAMLgetConf<std::string>(conf, "port", "/dev/ttyUSB0");
    baud    = tk::common::YAMLgetConf<int>(conf, "baud", 115200);
    lines   = tk::common::YAMLgetConf<std::vector<int>>(conf, "lines", std::vector<int>(0));
}

void 
Clock::start(timeStamp_t start)
{
    // SynchBox
    if (start == 0) {
        // open serial port
        if (!serial.init(port, baud)) {
            tkWRN("Cannot open communication with SynchBox.\n");
            synched = false;
            return;
        } else {
            synched = true;
        }

        //Send data
        std::string msg = "\\gse\n";
        serial.write(msg);
        std::string ts_0;
        while(ts_0.find("$GPRMC") == std::string::npos){
            std::cout<<ts_0<<std::endl;
            if(!serial.readLine(ts_0,'\n',5000))
                tkERR("Error reciving timestamp\n");
        }
        std::cout<<ts_0<<std::endl;
        int pos = 0;
        std::vector<std::string> seglist;
        while((pos = ts_0.find(',')) != std::string::npos){
            seglist.push_back(ts_0.substr(0, pos));
            ts_0.erase(0, pos+1);
        }
        msg = "\\gsd\n";
        serial.write(msg);
        serial.close();

        int hh = atoi(seglist[1].substr(0,2).c_str());
        int mm = atoi(seglist[1].substr(2,2).c_str());
        int ss = atoi(seglist[1].substr(4,2).c_str());
        int ms = atoi(seglist[1].substr(6,3).c_str());

        int day   = atoi(seglist[9].substr(0,2).c_str());
        int month = atoi(seglist[9].substr(2,2).c_str());
        int year  = atoi(seglist[9].substr(4,2).c_str())+2000;

        struct tm time = { 0 };
        time.tm_year = (int) year - 1900;
        time.tm_mon  = (int) month;
        time.tm_mday = (int) day;
        time.tm_hour = (int) hh + 1;
        time.tm_min  = (int) mm;
        time.tm_sec  = (int) ms;

        time_t t = mktime(&time) + 619315200;  /// Added 1024 weeks

        struct tm *tim = localtime(&t);

        timeStamp_t stamp = (t * 1e3);

        std::cout<<tim->tm_year + 1900<<"   "<<tim->tm_mon<<"   "<<tim->tm_mday<<"   "<<tim->tm_hour<<"   "<<tim->tm_min<<"   "<<stamp<<"\n";
        //t0 = std::stoi(ts_0.c_str());
    } else {
        t0      = start;
        synched = true;
    }
}

void 
Clock::stop()
{

}

timeStamp_t 
Clock::getTimeStamp(int frameCounter, int triggerLine)
{
    if (frameCounter == -1 || triggerLine == -1)
        return ::getTimeStamp();
    
    tkASSERT(synched == true);
    tkASSERT(triggerLine < lines.size(), "Out of bounds.\n");

    return (timeStamp_t) t0 + (frameCounter * 1.0f/lines[triggerLine]); 
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