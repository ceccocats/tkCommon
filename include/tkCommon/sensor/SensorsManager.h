#pragma once

#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawables.h"

#include "tkCommon/rt/Task.h"

#include "tkCommon/sensor/Sensor.h"
#include "tkCommon/sensor/SensorSpawner.h"
#include "tkCommon/sensor/LogManager.h"

namespace tk { namespace sensors {
    /**
     * @brief 
     * 
     */
    struct drawInfo{
        int                     id;
        bool                    locked;
        std::string             name;
        tk::data::sensorType    type;
        tk::gui::Drawable       *drawable;
    };

    /**
     * @brief 
     * 
     */
    class SensorsManager {
    public:
        // Sensors
        std::map<std::string,tk::sensors::Sensor*>  sensors;
        
        // LOG
        tk::sensors::LogManager*    logManager = nullptr;
        std::string                 logpath = "";

        // GUI
        tk::gui::Viewer*        viewer;
        tk::rt::Thread          viewerThread;
        std::vector<drawInfo>   drawables;

        /**
         * @brief 
         * 
         * @param conf 
         * @param logPath 
         * @param list 
         * @param viewer 
         * @return true 
         * @return false 
         */
        bool init(YAML::Node conf, const std::string &logPath = "", 
                  SensorSpawner *spawner = nullptr,  const std::string &list="", 
                  tk::gui::Viewer* viewer = nullptr);

        /**
         * @brief 
         * 
         */
        void start();

        /**
         * @brief 
         * 
         * @return true 
         * @return false 
         */
        bool close();

        /**
         * @brief 
         * 
         * @param folderPath 
         */
        void startRecord(const std::string &folderPath);

        /**
         * @brief 
         * 
         */
        void stopRecord();

        /**
         * @brief 
         * 
         * @param s 
         * @return tk::sensors::Sensor* 
         */
        tk::sensors::Sensor* operator[](const std::string &s);

        /**
         * @brief Set the Replay Debug object
         * 
         * @param debug 
         */
        void setReplayDebug(bool debug);

        /**
         * @brief 
         * 
         * @param time 
         */
        void skipDebugTime(timeStamp_t time);

        /**
         * @brief When LogManger is in manual mode, wait until all sensors reached logTick
         * @param time ref logTick time
         */
        void waitSync(timeStamp_t time);

    private:
        /**
         * @brief 
         * 
         * @param args 
         * @return void* 
         */
        static void* dataViewer(void *args);
    };
}}
