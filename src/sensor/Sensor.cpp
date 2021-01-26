#include "tkCommon/sensor/Sensor.h"

namespace tk { namespace sensors {

void 
Sensor::start() {
    // start loop thread
    pthread_attr_init(&attr);
    pthread_create(&t0, &attr, &Sensor::loop_helper, this);
    readLoopStarted = true;
}

void* 
Sensor::loop_helper(void *context) {
    return ((tk::sensors::Sensor*)context)->loop(nullptr);
}

void* 
Sensor::loop(void *vargp) {

    while (senStatus != sensorStatus::STOPPING && senStatus != sensorStatus::ERROR) {
        if (!readFrame()) {
            tkERR("Error while reading\n");
            sleep(1);
            continue;
        }
        if(viewer != nullptr) {
            // TODO do something 
        }
    }

    pthread_exit(nullptr);
}

void 
Sensor::setViewer(tk::gui::Viewer *viewer) {
    this->viewer = viewer;
    // TODO add a drawable to viewer
}

tk::common::Tfpose 
Sensor::getTf(int id) {
    if(id >= tf.size())
        return tk::common::Tfpose::Identity();
    else
        return tf[id];
}

std::vector<tk::common::Tfpose> 
Sensor::getTfs() {
    return tf;
}

bool 
tk::sensors::Sensor::close() {
	if (senStatus == sensorStatus::OFFLINE)
		log->stop();

    if (senStatus == sensorStatus::RECORDING) {

        stopRecord();
    }

    senStatus = sensorStatus::STOPPING;

    if(readLoopStarted)
        pthread_join(t0, nullptr);

    pool.close();

    return true;
}

bool 
Sensor::readFrame() {
    // get data from pool
    bool    retval;
    int     idx;
    tk::data::SensorData* data = pool.add(idx);

    // check sensor status
    if (senStatus != sensorStatus::OFFLINE) {
        retval = readOnline(data);
    }else {         
        retval = readLog(data);
        if(retval)
            log->wait(data->header.stamp, info.synched);
        else
            info.synched = true;
    }

    if (poolEmpty && retval)
        poolEmpty = false;

    info.dataArrived++;

    // fill data header
    if(data->header.name != info.name)
        data->header.name = info.name;
    data->header.tf         = getTf();
    data->header.messageID  = info.dataArrived;

    // release pool
    pool.releaseAdd(idx);

    if(first_read) {
        first_read          = false;
        info.startingTime   = getTimeStamp();
    } else {

        timeStamp_t t   = getTimeStamp() - info.startingTime;
        info.readFps    = (float)info.dataArrived / (t/1000000.0);
    }
    
    return retval;
}

tk::sensors::sensorStatus 
Sensor::status(){
    return this->senStatus;
}

void
Sensor::release(const int idx) {
    pool.releaseGet(idx);
}
}}