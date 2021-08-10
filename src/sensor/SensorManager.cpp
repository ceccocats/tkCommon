#include "tkCommon/sensor/SensorsManager.h"

using namespace tk::sensors;
bool 
SensorsManager::init(YAML::Node aConf, const std::string &aLogPath, const std::string &aList, const bool &aGui) 
{    
    // LOG
    if (aLogPath != "") {
        this->mLogPath      = aLogPath;
        this->mLogManager   = new tk::sensors::LogManager();
        if (!mLogManager->init(this->mLogPath)) {
            tkERR("Error init logManger.\n");
            return false;
        }
        if (tk::gui::Viewer::getInstance()->isRunning()) //mGui not checked, replayinfo is useful
            tk::gui::Viewer::getInstance()->add(new tk::gui::ReplayInfo(mLogManager));
    }
    mGui = aGui;
    
    /*
    // SYNCH BOX
    if (conf["synch"].IsDefined()) {
        Clock::get().init(conf["synch"]);
    } else {
        tkWRN("No synch parameter defined, skipping synch.\n");
    }
    */

    // GUI
    if (mGui && tk::gui::Viewer::getInstance()->isRunning()) {
        //tk::gui::Viewer::getInstance()->add(new tk::gui::Grid());
        tk::gui::Viewer::getInstance()->add(new tk::gui::Axis());
    }

    // SPAWN SENSOR
    if (!spawn(aConf, aList)) {
        tkERR("Cannot spawn sensors.\n");
        return false;
    }

    return true;
}

void 
SensorsManager::start() 
{
    for (const auto &sensor: mSensors)
        sensor.second->start();
    //Clock::get().start();
}

bool 
SensorsManager::close() 
{
    for (const auto &sensor: mSensors)
        sensor.second->close();
    
    //if (Clock::get().synchronized())
    //    Clock::get().stop();
    return true;
}

void 
SensorsManager::startRecord(const std::string &aFolderPath) 
{
    for (const auto &sensor: mSensors)
        sensor.second->startRecord(aFolderPath);
    
    if (tk::gui::Viewer::getInstance()->isRunning())
        tk::gui::Viewer::getInstance()->add(new tk::gui::RecordInfo(aFolderPath));
}

void 
SensorsManager::stopRecord() 
{
    for (const auto &sensor: mSensors)
        sensor.second->stopRecord();
}

tk::sensors::Sensor* 
SensorsManager::operator[](const std::string &aString)
{
    if (mSensors.find(aString) != mSensors.end())
        return mSensors.find(aString)->second;
    else {
        tkERR("Cannot find \'"<<aString<<"\' sensor.\n");
        return nullptr;
    }
}

void 
SensorsManager::setReplayDebug(bool aDebug)
{
    if (mLogManager != nullptr) {
        mLogManager->manualTick = aDebug;
    } else {
        tkWRN("You are in real time mode, this is not possible\n");
    }
}

void 
SensorsManager::skipDebugTime(timeStamp_t aTime)
{
    if (mLogManager != nullptr) {
        mLogManager->setTick(mLogManager->getTick() + aTime);
    } else {
        tkWRN("You are in real time mode, this is not possible\n");
    }
}

void 
SensorsManager::waitSync(timeStamp_t aTime) 
{
    tkASSERT(mLogManager != nullptr);
    mLogManager->setTick(aTime);
    usleep(20000); // TODO: better sync

    while (true) {
        bool synched = true;
        for (const auto sensor : mSensors)
            synched &= sensor.second->info.synched;

        if (synched)
            break;

        usleep(1000);
    }
}