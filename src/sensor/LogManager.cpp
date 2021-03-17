#include "tkCommon/sensor/LogManager.h"

namespace tk { namespace sensors {
    bool 
    LogManager::init(const std::string folderPath) {
        this->logFolder = folderPath;
        this->readLog = !this->logFolder.empty();
        return this->readLog;
    }

    void 
    LogManager::setTick(timeStamp_t time) {
        tkASSERT(manualTick);
        logTick = time;
    }
        
    void* 
    LogManager::logTicker(void *args) {
        LogManager *self = (LogManager*) args;
        timeStamp_t ts = 100;

        std::chrono::microseconds inc_chrono {ts};
        std::chrono::system_clock::time_point currentpt = std::chrono::system_clock::now();
        while (tk::gui::Viewer::getInstance()->isRunning()) {
            if(self->manualTick) { 
                // manual tick (do nothing and restore start point in case it will switch automatic tick)
                usleep(500000);
                currentpt = std::chrono::system_clock::now();
            } else {
                // increment tick counter
                self->startStopMtx.lock();
                self->startStopMtx.unlock();
                timeStamp_t inc = float(ts)*self->speed;
                self->logTick += inc;
                currentpt += inc_chrono;
                std::this_thread::sleep_until(currentpt);
            }
        }

        pthread_exit(nullptr);
    }

    timeStamp_t 
    LogManager::wait(timeStamp_t time, bool &synched) {
        // no wait mode 
        if(noWait) {
            synched = true;
            return 0;
        }
        // invalid time, not synched
        if(time == 0) {
            synched = false;
            return 0;
        }

        // try to start (no problem if already started)
        startMtx.lock();
        start(time);
        startMtx.unlock();

        if(time < logTick) {
            // this read is before the target time, so we are a bit in hurry
            synched = false;
            return 0;
        }
        while(time > logTick && !stopped) {
            // this read is after the target time, keep calm
            synched = true;
            timeStamp_t delta = time - logTick;
            if(delta > 1000)    // limit sleep to avoid sleep to much
                delta = 1000;
            usleep(delta);
        }
        return time - logTick;
    }

    void 
    LogManager::stop() {
        stopped = true;
    }

    void 
    LogManager::start(timeStamp_t time) {
        // invalid timestamp or already started
        if(time == 0 || !readLog || logStarted)
            return;

        // set initial timestamp
        startStamp = time;
        logTick = startStamp + timeStamp_t(skipSec)*1000000;
        tkMSG("Initial time: "<<logTick<<"\n");
        
        // with manualTick "logTick" is manually updated, so we dont need the thread
        if(!manualTick) {
            pthread_t thread;
            void *self = this;
            pthread_create(&thread, nullptr, LogManager::logTicker, self);
        }
        logStarted = true;
    }
}}