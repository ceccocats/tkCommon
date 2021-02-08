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


namespace tk{ namespace sensors {

/** @brief   Sensor status class */
class SensorStatus {
    public:
        enum Value : uint8_t{
        ONLINE                  = 0,
        OFFLINE                 = 1,
        RECORDING               = 2,
        STOPPING                = 3,
        ERROR                   = 4
        };

        /**
         * @brief   method for convert id to semaphore status string name
         */
        std::string toString()
        {
            if(value == SensorStatus::ONLINE)                   return std::string{"online"};
            if(value == SensorStatus::OFFLINE)                  return std::string{"offline"};
            if(value == SensorStatus::RECORDING)                return std::string{"recording"};
            if(value == SensorStatus::STOPPING)                 return std::string{"stopping"};
            if(value == SensorStatus::ERROR)                    return std::string{"error"};
            return std::string{"type error"};
        }


        bool operator!=(SensorStatus::Value v) noexcept 
        {
            return v != value;
        }

        bool operator==(SensorStatus::Value v) noexcept 
        {
            return v == value;
        }

        void operator=(SensorStatus::Value v) noexcept 
        {
            value = v;
        }
    
    private:
        SensorStatus::Value value;
};

class SensorInfo{
    public:
        std::string     name;       /**< sensor name */
        float           version;    /**< programm version*/
        int             nSensors;
        int             dataArrived;
        bool            synched;    /**< tell if the sensor is synched with the log */
        tk::data::sensorType type;

        SensorInfo() 
        {
            name            = "?????";
            version         = 1.0f;
            nSensors        = 1;
            dataArrived     = 0;
            synched         = false;
            type            = tk::data::sensorType::NOT_SPEC;
        }

        /**
         * @brief 
         * 
         * @param i 
         */
        void operator=(SensorInfo i) noexcept {
            this->name              = i.name;
            this->version           = i.version;
            this->nSensors          = i.nSensors;
            this->dataArrived       = i.dataArrived;
            this->type              = i.type;
        }
};



/**
 * @brief Sensor class interface
 * 
 */
class Sensor {
    public:

        SensorInfo info; /**< */


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
        bool grab(const T* &data, int &idx, uint64_t timeout = 0) 
        {    
            if (poolEmpty)
                return false;
            
            if (timeout != 0) {     // locking
                data = dynamic_cast<const T*>(pool.get(idx, timeout));
            } else {                // non locking
                if (pool.newData(lastDataCounter)) {   // new data available
                    lastDataCounter = (uint32_t) pool.inserted;
                    data = dynamic_cast<const T*>(pool.get(idx)); 
                } else {                                    // no new data available
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
        SensorStatus status();

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
        
        SensorStatus    senStatus;  /** Sensor status */

        LogManager      *log = nullptr;

        tk::rt::DataPool pool;       /**< Data pool */
        int              poolSize;   /**< Size of data pool */
        bool             poolEmpty;        

        std::vector<tk::common::Tfpose> tf;         /**< Sensor tf */
        
        /**
         * @brief Method that init the sensor
         * 
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>> 
        bool initSensor(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) {
            // get class name
            this->info.name             = name;
            this->info.dataArrived      = 0;
            this->log                   = log;
            this->lastDataCounter       = 0;

            // check if paths passed are correct
            if (!conf) {
                tkERR("No sensor configuration in yaml\n");
                return false;
            }

            // read tf from configuration file
            if (conf["tf"].IsDefined()) {
                this->tf = tk::common::YAMLreadTf(conf["tf"]);
            } else {
                this->tf.resize(1);
                this->tf[0] = Eigen::Isometry3f::Identity();
            }

            // get configuration params
            this->poolSize = tk::common::YAMLgetConf<int>(conf, "pool_size", 2);

            // set sensor status
            if(this->log == nullptr)
                this->senStatus = SensorStatus::ONLINE;
            else 
                this->senStatus = SensorStatus::OFFLINE;

            // init pool
            if(this->poolSize < 1) {
                tkWRN("You tried to set poolSize to a negative value, resetted to 1.")
                this->poolSize = 1;
            }
            this->poolEmpty = true;
            this->pool.init<T>(this->poolSize);
            int idx;
            for (int i = 0; i < this->poolSize; i++) {
                auto data = dynamic_cast<T*>(this->pool.add(idx));
                
                data->header.name = name;
                data->header.tf = getTf();

                this->pool.releaseAdd(idx);
            }
            

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
         * @brief Method that read the sensor data 
         *
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        bool readFrame();

        
    
    private:
        std::mutex  readMtx;
        uint32_t    lastDataCounter;

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
