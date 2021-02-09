#pragma once
#include <thread>
#include "tkCommon/common.h"

extern bool gRun;
namespace tk { namespace sensors {

    class LogManager {
    public:
        std::string logFolder  = "";      /**< log folder to read */
        
        // configs
        bool  noWait     = false; /**< skip wait if true */
        bool  manualTick = false; /**< autoTick if false, manual tick with setTick() if true */
        int   skipSec    = 0;     /**< skip n seconds on start */
        float speed      = 1.0;   /**< autoTick speed, time speed multiplier */
        bool  stopped    = false;

        /**
         * @brief Init the log manager, remember to set the configs before or right after init, if you need it
         * @param folderPath Log path
         */
        bool init(const std::string folderPath);

        /**
         * @brief Manually set tick time
         * @param time 
         */
        void setTick(timeStamp_t time);

        /**
         * @brief get current tick
         */
        timeStamp_t getTick();

        /**
         * @brief auto-ticker thread
         */
        static void *logTicker(void *args);

        /**
         * @brief This is called by the sensor read, it waits to synch with other sensors 
         * @param time Data timestamps to be synced with logTick
         * @param synched update the sensor synch status
         */
        timeStamp_t wait(timeStamp_t time, bool &synched);

        /**
         * @brief Method to stop the replay.
         */
        void stop();

    private:
        void start(timeStamp_t time);

        bool        readLog    = false;   /**< read or not from log file */
        timeStamp_t logTick    = 0;       /**< timestamp cursor in log */
        timeStamp_t startStamp = 0;       /**< start timestamp in log */
        bool        logStarted = false;   /**< true if log is started */
        std::mutex  startMtx;
    };
}}