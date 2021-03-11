#pragma once

#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawables.h"

#include "tkCommon/sensor/Sensor.h"
#include "tkCommon/sensor/LogManager.h"

namespace tk { namespace sensors {
    /**
     * @brief   SensorsManager is a wrapper class around the Sensor class 
     *          capable of handling multiple sensors at the same time.
     */
    class SensorsManager {
    public:
        // Sensors
        std::map<std::string,tk::sensors::Sensor*>  mSensors;
        
        // LOG
        tk::sensors::LogManager*    mLogManager  = nullptr;
        std::string                 mLogPath     = "";

        /**
         * @brief   Spawn and initilize sensors.
         * 
         * @param conf      configuration file.
         * @param logPath   path to a recordings folder to be used for replay.
         * @param list      list of sensor names to be spawned.
         * @param viewer    pointer to viewer object.
         * @return true     if no error occours.
         * @return false    if an error occours.
         */
        bool init(YAML::Node aConf, const std::string &aLogPath = "", 
                  const std::string &aList="");

        /**
         * @brief   start all spawned sensors reading thread and an internal thread to update the viewer.
         */
        void start();

        /**
         * @brief   close all spawned sensors and the internal thread to update the viewer.
         * 
         * @return true     if no error occours.
         * @return false    if an error occours.
         */
        bool close();

        /**
         * @brief   start recording for all spawned sensor.
         * 
         * @param folderPath    path to a folder where the sensor will create his save file.
         */
        void startRecord(const std::string &aFolderPath);

        /**
         * @brief   stop recording for all spawned sensor.
         */
        void stopRecord();

        /**
         * @brief   method to directly acces to spawned sensors.
         * 
         * @param s     string, must be the same passed inside list parameter on init method.
         * @return tk::sensors::Sensor*     reference to Sensor
         */
        tk::sensors::Sensor* operator[](const std::string &aString);

        /**
         * @brief   Set the Replay Debug object
         * 
         * @param debug 
         */
        void setReplayDebug(bool aDebug);

        /**
         * @brief   Skip time.
         * 
         * @param time  Amount of time to be skipped.
         */
        void skipDebugTime(timeStamp_t aTime);

        /**
         * @brief   When LogManger is in manual mode, wait until all sensors reached logTick
         * @param time ref logTick time
         */
        void waitSync(timeStamp_t aTime);

    protected:
        /**
         * @brief   Spawn sensors.
         * 
         * @param conf      configuration file.
         * @param list      list of sensor names to be spawned.
         * @return true     if no error occours.
         * @return false    if an error occours.
         */
        virtual bool spawn(YAML::Node aConf, const std::string &aList="") = 0;
    };
}}
