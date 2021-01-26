#include "tkCommon/sensor/SensorsManager.h"

namespace tk { namespace sensors {
    bool 
    SensorsManager::init(YAML::Node conf, const std::string &logPath, const std::string &list, tk::gui::Viewer* viewer) 
    {    
        // LOG
        if (logPath != "") {
            logpath     = logPath;
            logManager  = new tk::sensors::LogManager();
            if (!logManager->init(logpath)) {
                tkERR("Error init logManger.\n");
                return false;
            }
        }
        
        // SPAWN SENSOR
        if (!spawn(conf, list)) {
            tkERR("Cannot spawn sensors.\n");
            return false;
        }

        // GUI
        this->viewer = viewer;
        if (viewer!= nullptr) {
            
            for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it) {
                if (it->second->info.type == tk::data::sensorType::LIDAR) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Cloud4f(it->second->info.name)});
                } else if (it->second->info.type == tk::data::sensorType::GPSIMU) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::GpsImu()});
                } else if (it->second->info.type == tk::data::sensorType::CAMDATA) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Image(it->second->info.nSensors, it->second->info.name)});
                }
            }

            if (!viewerThread.init(dataViewer,this)) {
                tkERR("Cannot start dataViewer thread\n");
                return false;
            }
        }

        return true;
    }

    void 
    SensorsManager::start() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->start();
    }

    bool 
    SensorsManager::close() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->close();

        viewerThread.join();
    }

    void 
    SensorsManager::startRecord(const std::string &folderPath) 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->startRecord(folderPath);
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
        return sensors.find(s)->second;
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

    void* 
    SensorsManager::dataViewer(void *args) {
        auto self = (SensorsManager *)args;

        self->viewer->add(new tk::gui::Grid());
        self->viewer->add(new tk::gui::Axis());

        for (auto drw : self->drawables) {
            self->viewer->add(drw.drawable);
        }

        tk::rt::Task t;
        t.init(16000);
        while (self->viewer->isRunning()){
            for (int i = 0; i < self->drawables.size(); i++){
                auto drw = &(self->drawables[i]);
                if (drw->type == tk::data::sensorType::GPSIMU) {
                    auto ref = (tk::gui::GpsImu*)drw->drawable;
                    if (drw->locked){
                        if (ref->update == false){
                            self->sensors[drw->name]->release(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::GpsImuData* d;
                        if (self->sensors[drw->name]->grab<tk::data::GpsImuData>(d,drw->id)) {
                            tk::data::GpsImuData* a = (tk::data::GpsImuData*)d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if(drw->type == tk::data::sensorType::LIDAR) {
                    auto ref = (tk::gui::Cloud4f*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::CloudData* d;
                        if (self->sensors[drw->name]->grab<tk::data::CloudData>(d,drw->id)) {
                            tk::data::CloudData* a = (tk::data::CloudData*)d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if(drw->type == tk::data::sensorType::CAMDATA) {
                    auto ref = (tk::gui::Image*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::VectorData<tk::data::ImageData>* d;
                        if (self->sensors[drw->name]->grab(d,drw->id)) {
                            auto a = (tk::data::VectorData<tk::data::ImageData>*)d;
                            for(auto &t : a->data){
                                if(t.isGPU) t.data.synchCPU();
                            }
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                }
            }
            t.wait();
        }
    }
}}