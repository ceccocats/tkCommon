#include "tkCommon/sensor/SensorsManager.h"

namespace tk { namespace sensors {
    bool 
    SensorsManager::init(YAML::Node conf, const std::string &logPath, const std::string &list, tk::gui::Viewer* viewer) 
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
        
        // SYNCH BOX
        if (conf["synch"].IsDefined()) {
            Clock::get().init(conf["synch"]);
        } else {
            tkWRN("No synch parameter defined, skipping synch.\n");
        }

        // SPAWN SENSOR
        if (!spawn(conf, list)) {
            tkERR("Cannot spawn sensors.\n");
            return false;
        }

        // GUI
        if (viewer != nullptr) {
            this->viewer = viewer;
            for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it) {
                if (it->second->info.type == tk::data::DataType::CLOUD) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Cloud4f(it->second->info.name)});
                } else if (it->second->info.type == tk::data::DataType::GPSIMU) {
                    drawables.push_back({0, false, it->second->info.name, tk::data::DataType::GPS, new tk::gui::Gps(it->second->info.name + "_gps")});
                    drawables.push_back({0, false, it->second->info.name, tk::data::DataType::IMU, new tk::gui::Imu(it->second->info.name + "_imu")});
                } else if (it->second->info.type == tk::data::DataType::IMAGE) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Image(it->second->info.nSensors, it->second->info.name)});
                } else if (it->second->info.type == tk::data::DataType::GPS) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Gps(it->second->info.name)});
                } else if (it->second->info.type == tk::data::DataType::RADAR) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Radar(it->second->info.name)});
                } else if (it->second->info.type == tk::data::DataType::STEREO) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Image(2, it->second->info.name)});
                } else if (it->second->info.type == tk::data::DataType::IMU) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Imu()});
                } else if (it->second->info.type == tk::data::DataType::CAN) {
                    drawables.push_back({0, false, it->second->info.name, it->second->info.type, new tk::gui::Can()});
                }
            }
        }

        return true;
    }

    void 
    SensorsManager::start() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->start();
        
        //Clock::get().start();

        if (this->viewer != nullptr)
            viewerThread.init(dataViewer,this);
    }

    bool 
    SensorsManager::close() 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->close();
        
        if (Clock::get().synchronized())
            Clock::get().stop();
        
        if (this->viewer != nullptr)
            viewerThread.join();
    }

    void 
    SensorsManager::startRecord(const std::string &folderPath) 
    {
        for (std::map<std::string,tk::sensors::Sensor*>::iterator it = sensors.begin(); it!=sensors.end(); ++it)
            it->second->startRecord(folderPath);
        
        viewer->add(new tk::gui::RecordInfo(folderPath));
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
                if(drw->type == tk::data::DataType::CLOUD) {
                    auto ref = (tk::gui::Cloud4f*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release<tk::data::CloudData>(drw->id);
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
                } else if(drw->type == tk::data::DataType::IMAGE) {
                    auto ref = (tk::gui::Image*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release<tk::data::VectorData<tk::data::ImageData>>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::VectorData<tk::data::ImageData>* d;
                        if (self->sensors[drw->name]->grab<tk::data::VectorData<tk::data::ImageData>>(d,drw->id)) {
                            auto a = (tk::data::VectorData<tk::data::ImageData>*)d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if(drw->type == tk::data::DataType::GPS) {
                    auto ref = (tk::gui::Gps*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release<tk::data::GpsData>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::GpsData* d;
                        if (self->sensors[drw->name]->grab<tk::data::GpsData>(d,drw->id)) {
                            auto a = (tk::data::GpsData*) d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if(drw->type == tk::data::DataType::RADAR) {
                    auto ref = (tk::gui::Radar*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release<tk::data::RadarData>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::RadarData* d;
                        if (self->sensors[drw->name]->grab<tk::data::RadarData>(d,drw->id)) {
                            auto a = (tk::data::RadarData*) d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if(drw->type == tk::data::DataType::STEREO) {
                    auto ref = (tk::gui::Image*)drw->drawable;
                    if (drw->locked) {
                        if (ref->update == false) {
                            self->sensors[drw->name]->release<tk::data::StereoData>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::StereoData* d;
                        if (self->sensors[drw->name]->grab<tk::data::StereoData>(d,drw->id)) {
                            auto a = (tk::data::StereoData*)d;
                            ref->updateRef(0,&a->data);
                            if(!a->color.empty()){
                                ref->updateRef(1, &a->color);
                            }
                            drw->locked = true;
                        }
                    }
                } else if (drw->type == tk::data::DataType::IMU) {
                    auto ref = (tk::gui::Imu*)drw->drawable;
                    if (drw->locked){
                        if (ref->update == false){
                            self->sensors[drw->name]->release<tk::data::ImuData>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::ImuData* d;
                        if (self->sensors[drw->name]->grab<tk::data::ImuData>(d,drw->id)) {
                            tk::data::ImuData* a = (tk::data::ImuData*)d;
                            ref->updateRef(a);
                            drw->locked = true;
                        }
                    }
                } else if (drw->type == tk::data::DataType::CAN) {
                    auto ref = (tk::gui::Can*)drw->drawable;
                    if (drw->locked){
                        if (ref->update == false){
                            self->sensors[drw->name]->release<tk::data::CanData>(drw->id);
                            drw->locked = false;
                        }
                    } else {
                        const tk::data::CanData* d;
                        if (self->sensors[drw->name]->grab<tk::data::CanData>(d,drw->id)) {
                            tk::data::CanData* a = (tk::data::CanData*)d;
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