/**
 * @file    Sensor.h
 * @author  Luca Bartoli, Fabio Bagni, Gatti Francesco, Bosi Massimiliano
 * @brief   Sensor template class
 * @version 0.1
 * @date    2019-10-02
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once
#include <fstream>
#include <mutex>
#include <pthread.h>

#include "tkCommon/common.h"
#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/SensorData.h"
#include "tkCommon/sensor/LogManager.h"
#include "tkCommon/gui/Viewer.h"


namespace tk{ namespace sensors{

/** @brief   Sensor status class */
class sensorStatus{
    public:
        enum Value : uint8_t{
        ONLINE                  = 0,
        OFFLINE                 = 1,
        RECORDING               = 2,
        RECORDING_AND_READING   = 3,
        STOPPING                = 4,
        ERROR                   = 5
        };

        /**
         * @brief   method for convert id to semaphore status string name
         */
        std::string toString(){
            if(value == sensorStatus::ONLINE)                   return std::string{"online"};
            if(value == sensorStatus::OFFLINE)                  return std::string{"offline"};
            if(value == sensorStatus::RECORDING)                return std::string{"recording"};
            if(value == sensorStatus::RECORDING_AND_READING)    return std::string{"recording and reading"};
            if(value == sensorStatus::STOPPING)                 return std::string{"stopping"};
            if(value == sensorStatus::ERROR)                    return std::string{"error"};
            return std::string{"type error"};
        }

        /**
         * @brief 
         * 
         * @param v 
         * @return true 
         * @return false 
         */
        bool operator!=(sensorStatus::Value v) noexcept {
            return v != value;
        }

        /**
         * @brief 
         * 
         * @param v 
         * @return true 
         * @return false 
         */
        bool operator==(sensorStatus::Value v) noexcept {
            return v == value;
        }

        /**
         * @brief 
         * 
         * @param v 
         */
        void operator=(sensorStatus::Value v) noexcept {
            value = v;
        }
    
    private:
        sensorStatus::Value value;
};

class sensorInfo{
    public:
        std::string     name;           /**< sensor name */
        tk::data::sensorType type;

        float           version;        /**< programm version*/
        uint16_t        dataArrived;    /**< number of data arrived form the sensor*/
        uint16_t        dataReaded;     /**< number of data readed from the sensor*/
        timeStamp_t     startingTime;   /**< sensor starting time*/
        float           readFps;        /**< sensor data fps */
        int             nSensors = 1;

        bool            synched = false; /**< tell if the sensor is synched with the log */

        /**
         * @brief 
         * 
         * @param i 
         */
        void operator=(sensorInfo i) noexcept {
            this->name          = i.name;
            this->version       = i.version;
            this->dataArrived   = i.dataArrived;
            this->dataReaded    = i.dataReaded;
            this->startingTime  = i.startingTime;
            this->readFps       = i.readFps;
            this->type          = i.type;
        }
};



/**
 * @brief Sensor class interface
 * 
 */
class Sensor {
    public:

        sensorInfo info; /**< */


        /**
         * Init sensor class, that also call the son initSensor
         * @param conf  configuration file
         * @param name  unique string sensor name
         * @param log   pointer to LogManager instance
         * 
         * @return true     Successful init
         * @return false    Unsuccessful init
         */
        virtual bool init(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) = 0;

        /**
         * @brief Start internal thread that read the sensor and fills the internal pool
         * 
         */
        void start();

        /**
         * @brief Extract last element from the pool. You must call before the readOnThread() method.
         * 
         * @param data      returned data
         * @param timeout   grab attend timeout
         * 
         * @return true     data is aviable
         * @return false    data is not aviable
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        bool grab(const T* &data, int &idx, uint64_t timeout = 0) {
            if (poolEmpty)
                return false;
            
            if (timeout != 0) {     // locking
                data = dynamic_cast<const T*>(pool.get(idx, timeout));
            } else {                // non locking
                if (pool.newData(info.dataReaded)) {    // new data available
                    info.dataReaded = pool.inserted;
                    data = dynamic_cast<const T*>(pool.get(idx)); 
                } else {                                // no new data available
                    data = nullptr;
                }
            }
            return (data != nullptr)?true:false;
        }

        /**
         * @brief 
         * 
         */
        void release(const int idx);

        /**
         * @brief Method that write sensor data (virtual)
         * 
         * @param  data     pointer to generic sensorData that you need to write
         * @return true     successful writing
         * @return false    unsuccessful writing
         */
        virtual bool write(tk::data::SensorData* data) = 0;

        /**
         * @brief Method that close the class (Virtual)
         * 
         * @return true     successful closing
         * @return false    unsuccessful closing
         */
        virtual bool close();

        /**
         * @brief Method that start the recording of sensor data (Virtual)
         * 
         * @param fileName string 
         * @return true     successful starting recording
         * @return false    unsuccessful starting recording
         */
        virtual bool startRecord(const std::string &fileName) = 0;
        
        /**
         * @brief Method that stop the recording of sensor data (Virtual)
         * 
         * @return true     successful stop recording
         * @return false    unsuccessful stop recording
         */
        virtual bool stopRecord() = 0;

        /**
         * @brief Status of the sensor
         *
         * @return sensor   status
         */  
        sensorStatus status();

        /**
         * @brief Method for set the sensor viewer
         * 
         * @param viewer viewer
         */
        void setViewer(tk::gui::Viewer *viewer);

        /**
         * @brief Get the Tf object
         * 
         * @param id tf id
         * @return tk::common::Tfpose 
         */
        tk::common::Tfpose getTf (const int id = 0);

        /**
         * @brief Get the Tfs object
         * 
         * @return std::vector<tk::common::Tfpose> 
         */
        std::vector<tk::common::Tfpose> getTfs();

    protected:
    
        // threads attributes
        bool            readLoopStarted = false; // true if the read loop is started
        pthread_t       t0;
        pthread_attr_t  attr;
        
        sensorStatus    senStatus;  /** Sensor status */

        tk::rt::DataPool pool;       /**< Data pool */
        int              poolSize;   /**< Size of data pool */
        bool             poolEmpty;
        
        std::vector<tk::common::Tfpose>     tf;         /**< Sensor tf */
        LogManager                          *log    = nullptr;
        bool                                first_read; /**< do first read, for fps calc */

        tk::gui::Viewer                     *viewer = nullptr; /**< viewer */

        /**
         * @brief Method that init the sensor
         * 
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>> 
        bool initSensor(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) {
            // get class name
            info.name           = name;
            info.version        = 0.1;
            info.dataArrived    = info.dataReaded = 0;
            first_read          = true;

            // check if paths passed are correct
            if (!conf) {
                tkERR("No sensor configuration in yaml\n");
                return false;
            }

            // read tf from configuration file
            if (conf["tf"].IsDefined()) {
                tf = tk::common::YAMLreadTf(conf["tf"]);
            } else {
                tf.resize(1);
                tf[0] = Eigen::Isometry3f::Identity();
            }

            //1.3 Get configuration params
            poolSize                = tk::common::YAMLgetConf<int>(conf, "pool_size", 2);
            bool readWhileRecord    = tk::common::YAMLgetConf<bool>(conf, "read_while_record", false);
            this->log               = log;

            //1.4 Set sensor status
            if(log == nullptr){
                if(readWhileRecord){
                    senStatus = sensorStatus::RECORDING_AND_READING;
                } else {
                    senStatus = sensorStatus::ONLINE;
                }
            } else {
                senStatus = sensorStatus::OFFLINE;
            }

            // init pool data
            if(poolSize < 1)
                poolSize = 1;
            
            pool.init<T>(poolSize);
            int idx;
            for (int i = 0; i < poolSize; i++) {
                T* data = dynamic_cast<T*>(pool.add(idx));
                
                data->header.name = name;
                data->header.tf = getTf();

                pool.releaseAdd(idx);
            }
            poolEmpty = true;

            return true;
        }
        
        /**
         * @brief Method that read the sensor data from online stream (Virtual)
         * 
         * @param data 
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        virtual bool readOnline(tk::data::SensorData* data) = 0;
        
        /**
         * @brief Method that read the sensor data from log file stream (Virtual)
         * 
         * @param data 
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        virtual bool readLog(tk::data::SensorData* data) = 0;

        /**
         * @brief Useful method for make some stuff before get data
         * 
         * @param data 
         */
        virtual void prepareData (tk::data::SensorData* data) = 0;

        /**
         * @brief Method that read the sensor data 
         *
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        bool readFrame();

        
    
    private:
        /**
         * @brief Method for launch start thread
         * 
         * @param context 
         * @return void* 
         */
        static void* loop_helper(void *context);

        /**
         * @brief Internal thread function to read the sensor
         * 
         * @param   vargp   class reference
         * @return  null    null
         */
        void* loop(void *vargp);                
};
}}
