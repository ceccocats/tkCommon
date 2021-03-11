#include "tkCommon/sensor/SensorsManager.h"

namespace tk { namespace sensors {
    bool 
    SensorsManager::init(YAML::Node conf, const std::string &logPath, const std::string &list) 
    {    
        // LOG
        if (logPath != "") {
            this->logPath       = logPath;
            this->logManager    = new tk::sensors::LogManager();
            if (!logManager->init(this->logPath)) {
                tkERR("Error init logManger.\n");
                return false;
            }
        }
        
        /*
        // SYNCH BOX
        if (conf["synch"].IsDefined()) {
            Clock::get().init(conf["synch"]);
        } else {
            tkWRN("No synch parameter defined, skipping synch.\n");
        }
        */

        // SPAWN SENSOR
        if (!spawn(conf, list)) {
            tkERR("Cannot spawn sensors.\n");
            return false;
        }

        return true;
    }

    void 
    SensorsManager::start() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->start();
        
        //Clock::get().start();
    }

    bool 
    SensorsManager::close() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->close();
        
        //if (Clock::get().synchronized())
        //    Clock::get().stop();
    }

    void 
    SensorsManager::startRecord(const std::string &folderPath) 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->startRecord(folderPath);
        
        if (tk::gui::Viewer::getInstance()->isRunning())
            tk::gui::Viewer::getInstance()->add(new tk::gui::RecordInfo(folderPath));
    }

    void 
    SensorsManager::stopRecord() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->stopRecord();
    }

    tk::sensors::Sensor* 
    SensorsManager::operator[](const std::string &s)
    {
        if (sensors.find(s) != sensors.end())
            return sensors.find(s)->second;
        else {
            tkERR("Cannot find \'"<<s<<"\' sensor.\n");
            return nullptr;
        }
    }

    void 
    SensorsManager::setReplayDebug(bool debug)
    {
        if (logManager != nullptr) {
            logManager->manualTick = debug;
        } else {
            tkWRN("You are in real time mode, this is not possible\n");
        }
    }

    void 
    SensorsManager::skipDebugTime(timeStamp_t time)
    {
        if (logManager != nullptr) {
            logManager->setTick(logManager->getTick() + time);
        } else {
            tkWRN("You are in real time mode, this is not possible\n");
        }
    }

    void 
    SensorsManager::waitSync(timeStamp_t time) 
    {
        tkASSERT(logManager != nullptr);
        logManager->setTick(time);
        usleep(20000); // TODO: better sync

        while (true) {
            bool synched = true;
            for (auto s : sensors)
                synched &= s.second->info.synched;

            if (synched)
                break;

            usleep(1000);
        }
    }
}}