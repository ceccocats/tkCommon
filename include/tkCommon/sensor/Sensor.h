#pragma once

#include <mutex>
#include <thread>
#include <map>
#include <utility>

#include "tkCommon/common.h"
#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/SensorData.h"
#include "tkCommon/sensor/LogManager.h"
#include "tkCommon/communication/serial/SerialPort.h"


namespace tk{ namespace sensors {

class Clock {
    public:
        ~Clock();
        Clock(Clock const&) = delete;
        void operator=(Clock const&) = delete;
        
        void init(const YAML::Node conf);
        void start(timeStamp_t start = 0);
        void stop();

        bool synchronized() {
            return synched || serial.isOpen();
        }

        static Clock& get()
        {
            static Clock instance;
            return instance;
        }

        timeStamp_t getTimeStamp(int frameCounter = -1, int triggerLine = -1);
    private:
        int                 timezone;
        timeStamp_t         t0;
        bool                synched;
        std::string         port;
        int                 baud;
        bool                mRun;
        std::vector<int>    lines;
        std::thread *th;
        tk::communication::SerialPort serial;

        Clock();
};

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
        std::map<uint32_t, int> dataArrived;    /**< incremental counter */
        int             triggerLine;
        bool            synched;        /**< tell if the sensor is synced with the log */
        tk::data::DataType type;      /**< type of the sensor, used for visualization */

        /**
         * @brief Construct a new SensorInfo object
         */
        SensorInfo() 
        {
            name            = "???";
            version         = 1.0f;
            nSensors        = 1;
            synched         = false;
            type            = tk::data::DataType::NOT_SPEC;
            triggerLine     = -1;
        }

        void operator=(SensorInfo i) noexcept {
            this->name              = i.name;
            this->version           = i.version;
            this->nSensors          = i.nSensors;
            this->dataArrived       = i.dataArrived;
            this->synched           = i.synched;
            this->type              = i.type;
            this->triggerLine       = i.triggerLine;
        }
};

struct SensorPool_t {
    tk::rt::DataPool    pool;
    int                 size;
    uint32_t            lastDataCounter;
    bool                empty;
};


/**
 * @brief Sensor class interface
 */
class Sensor {
    public:
        SensorInfo info;    /**< sensor info */

        /**
         * @brief   Method that init the sensor
         * 
         * @param conf  configuration file
         * @param name  name of the sensor
         * @param log   pointer to LogManager instance
         * @return true     successful init
         * @return false    unsuccessful init
         */
        virtual bool init(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) final;

        /**
         * @brief   Start internal thread that read the sensor and fills the internal pool
         */
        virtual void start() final;

        /**
         * @brief 
         * 
         * @param data 
         * @return true 
         * @return false 
         */
        virtual bool read(tk::data::SensorData* data) final;

        /**
         * @brief   Method that close the class.
         * 
         * @return true     successful closing
         * @return false    unsuccessful closing
         */
        virtual bool close() final;

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
        SensorStatus    senStatus;  /** Sensor status */
        LogManager      *log;       /** Log Manager reference */
        std::map<uint32_t, SensorPool_t*> pool;   /**< data pool */   
        int poolSize;
        std::vector<tk::common::Tfpose>             tf;     /**< Sensor TF */

        /**
         * @brief   Init sensor class, must be implemented by child.
         * 
         * @param conf      configuration file
         * @param name      unique string representing sensor name
         * @param log       pointer to LogManager instance
         * @return true     Successful init
         * @return false    Unsuccessful init
         */
        virtual bool initChild(const YAML::Node conf, const std::string &name, LogManager *log = nullptr) = 0;

        /**
         * @brief 
         * 
         * @return true 
         * @return false 
         */
        virtual bool closeChild() = 0;
        
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

        /**
         * @brief 
         * 
         * @tparam T 
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        void addPool(int sensorID = 0)
        {
            tk::sensors::SensorPool_t *sPool = new tk::sensors::SensorPool_t;
            sPool->empty            = true;
            sPool->size             = poolSize;
            sPool->lastDataCounter  = -1;
            sPool->pool.init<T>(sPool->size);

            int idx;
            for (int i = 0; i < sPool->size; i++) {
                auto data = dynamic_cast<T*>(sPool->pool.add(idx));
                
                data->header.name       = info.name + "_" + tk::data::ToStr(T::type);
                data->header.type       = T::type;
                data->header.sensorID   = sensorID;
                //data->header;

                sPool->pool.releaseAdd(idx);
            }
            
            // add pool
            pool.insert(std::pair<uint32_t, tk::sensors::SensorPool_t*>(uint32_t(T::type) + uint32_t(sensorID*1000), sPool));
            info.dataArrived.insert(std::pair<uint32_t, int>(uint32_t(T::type), 0));
        }

    private:
        // threads attributes
        bool                        readLoopStarted;    /**< true if the read loop is started */
        std::vector<std::thread>    readingThreads;

        /**
         * @brief   Internal reading thread.
         * 
         * @param   vargp   class reference
         * @return  void*   null
         */
        void loop(uint32_t type);

    public:
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
        bool grab(const T* &data, int &idx, uint64_t timeout = 0, int sensorID = 0) 
        {    
            // check if the passed template is the same as the pointer
            if (T::type != data->type) {
                tkERR("Type mismatch between template ("<<tk::data::ToStr(T::type)<<" and passed pointer ("<<tk::data::ToStr(data->type)<<").\n");
                return false;
            }

            // check if template type is present inside pool
            std::map<uint32_t, SensorPool_t*>::iterator it = pool.find(uint32_t(T::type) + uint32_t(sensorID*1000)); 
            if (it == pool.end()) {
                tkERR("No pool present with this template type.\n");
                return false;
            }

            // check if pool is empty
            if (it->second->empty) {
                tkWRN("Pool empty.\n");
                return false;
            }
            
            // grab
            if (timeout != 0) {     // locking
                data = dynamic_cast<const T*>(it->second->pool.get(idx, timeout));
                if (data == nullptr)
                    tkWRN("Timeout.\n");
            } else {                // non locking
                if (it->second->pool.newData(it->second->lastDataCounter)) {   // new data available
                    it->second->lastDataCounter = (uint32_t) it->second->pool.inserted;
                    data = dynamic_cast<const T*>(it->second->pool.get(idx)); 
                } else {                                    // no new data available
                    data = nullptr;
                }
            }
            return (data != nullptr)?true:false;
        }

        /**
         * @brief   Release grabbed element from the pool, must be called after a grab().
         * 
         * @tparam T  
         * @param idx   index of the data returned from the pool, given by the grab() method.
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        bool release(const int idx)
        {
            // check if template type is present inside pool
            std::map<uint32_t, SensorPool_t*>::iterator it = pool.find(uint32_t(T::type)); 
            if (it == pool.end()) {
                tkERR("No pool present with this template type.\n");
                return false;
            }

            // release element
            it->second->pool.releaseGet(idx);
            return true;
        }                
};
}}
