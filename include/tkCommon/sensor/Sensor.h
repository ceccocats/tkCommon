#pragma once

#include <mutex>
#include <pthread.h>

#include "tkCommon/common.h"
#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/SensorData.h"
#include "tkCommon/sensor/LogManager.h"


namespace tk{ namespace sensors {

/**
 * @brief   SensorStatus class, is used to handle the status of a sensor. 
 */
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
         * @brief   method to convert enum value to string
         */
        std::string toString()
        {
            if(value == SensorStatus::ONLINE)       return std::string{"online"};
            if(value == SensorStatus::OFFLINE)      return std::string{"offline"};
            if(value == SensorStatus::RECORDING)    return std::string{"recording"};
            if(value == SensorStatus::STOPPING)     return std::string{"stopping"};
            if(value == SensorStatus::ERROR)        return std::string{"error"};
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
        std::string     name;           /**< sensor name */
        float           version;        /**< program version */
        int             nSensors;       /**< number of sensors handled */
        int             dataArrived;    /**< incremental counter */
        bool            synched;        /**< tell if the sensor is synced with the log */
        tk::data::sensorType type;      /**< type of the sensor, used for visualization */

        /**
         * @brief Construct a new SensorInfo object
         */
        SensorInfo() 
        {
            name            = "???";
            version         = 1.0f;
            nSensors        = 1;
            dataArrived     = 0;
            synched         = false;
            type            = tk::data::sensorType::NOT_SPEC;
        }

        void operator=(SensorInfo i) noexcept {
            this->name              = i.name;
            this->version           = i.version;
            this->nSensors          = i.nSensors;
            this->dataArrived       = i.dataArrived;
            this->synched           = i.synched;
            this->type              = i.type;
        }
};


/**
 * @brief Sensor class interface
 */
class Sensor {
    public:
        SensorInfo info;    /**< sensor info */

        /**
         * @brief   Init sensor class, must be implemented by child.
         * 
         * @param conf      configuration file
         * @param name      unique string representing sensor name
         * @param log       pointer to LogManager instance
         * @return true     Successful init
         * @return false    Unsuccessful init
         */
        virtual bool init(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) = 0;

        /**
         * @brief   Start internal thread that read the sensor and fills the internal pool
         */
        void start();

        /**
         * @brief   Extract last or newest element from the pool, based on timeout parameter value
         *          passed. Must be called after start() method.
         * 
         * @param data      returned data
         * @param idx       index of the data returned from the pool, used in release() method.
         * @param timeout   grab timeout
         * @return true     data is available
         * @return false    data is not available
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
         * @brief   Release grabbed element from the pool, must be called after a grab().
         * 
         * @param idx   index of the data returned from the pool, given by the grab() method.
         */
        void release(const int idx);

        /**
         * @brief   Method that write sensor data.
         * 
         * @param  data     pointer to generic sensorData that you need to write
         * @return true     successful writing
         * @return false    unsuccessful writing
         */
        virtual bool write(tk::data::SensorData* data) = 0;

        /**
         * @brief   Method that close the class.
         * 
         * @return true     successful closing
         * @return false    unsuccessful closing
         */
        virtual bool close();

        /**
         * @brief   Method that start the recording of sensor data.
         * 
         * @param fileName  string representing the folder path
         * @return true     successful starting recording
         * @return false    unsuccessful starting recording
         */
        virtual bool startRecord(const std::string &fileName) = 0;
        
        /**
         * @brief   Method that stop the recording of sensor data.
         * 
         * @return true     successful stop recording
         * @return false    unsuccessful stop recording
         */
        virtual bool stopRecord() = 0;

        /**
         * @brief   Status of the sensor
         *
         * @return sensor   status
         */  
        SensorStatus status();

        /**
         * @brief   Get the TF object.
         * 
         * @param id tf id
         * @return tk::common::Tfpose 
         */
        tk::common::Tfpose getTf (const int id = 0);

        /**
         * @brief   Get vector of TF objects.
         * 
         * @return std::vector<tk::common::Tfpose> 
         */
        std::vector<tk::common::Tfpose> getTfs();

    protected:
    
        // threads attributes
        bool            readLoopStarted = false; /**< true if the read loop is started */
        pthread_t       t0;
        pthread_attr_t  attr;
        
        SensorStatus    senStatus;  /** Sensor status */

        LogManager      *log = nullptr;

        tk::rt::DataPool pool;      /**< data pool */
        int              poolSize;  /**< size of data pool */
        bool             poolEmpty; /**< true if no data has been added to the pool yet */        

        std::vector<tk::common::Tfpose> tf; /**< Sensor TF */

        /**
         * @brief   Method that init the sensor
         * 
         * @param conf  configuration file
         * @param name  name of the sensor
         * @param log   pointer to LogManager instance
         * @return true     successful init
         * @return false    unsuccessful init
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
         * @brief   Method that read sensor data from online stream
         * 
         * @param data      pointer to generic sensorData that will be filled.
         * @return true     successful read.
         * @return false    unsuccessful read.
         */
        virtual bool readOnline(tk::data::SensorData* data) = 0;
        
        /**
         * @brief   Method that read sensor data from log file stream
         * 
         * @param data      pointer to generic sensorData that will be filled.
         * @return true     successful read.
         * @return false    unsuccessful read.
         */
        virtual bool readLog(tk::data::SensorData* data) = 0;

        /**
         * @brief   Method that read the sensor data 
         *
         * @return true     Successful read
         * @return false    Unsuccessful read
         */
        bool readFrame();

    private:
        uint32_t    lastDataCounter;    /**< Last data counter inserted in the pool */

        /**
         * @brief   Method to launch internal reading thread
         * 
         * @param context 
         * @return void* 
         */
        static void* loop_helper(void *context);

        /**
         * @brief   Internal reading thread.
         * 
         * @param   vargp   class reference
         * @return  void*   null
         */
        void* loop(void *vargp);                
};
}}
