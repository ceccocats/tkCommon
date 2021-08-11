#pragma once

#include <mutex>
#include <thread>
#include <map>
#include <utility>

#include "tkCommon/common.h"
#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/SensorData.h"
#include "tkCommon/sensor/LogManager.h"
#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/drawables/Drawables.h"
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


struct SensorPool_t {
    tk::rt::DataPool    pool;
    int                 size;
    bool                empty;
    tk::gui::DataDrawable   *drw;
    
    tk::data::HeaderData header;
    std::vector<timeStamp_t> lastStamps; /**< vector of last N timestamps (used to compute FPS) */
    int lastStampsIdx = 0;
};

typedef std::pair<tk::data::DataType, int> sensorKey;

class SensorInfo{
    public:
        std::string     name;           /**< sensor name */
        float           version;        /**< program version */
        int             nSensors;       /**< number of sensors handled */
        std::map<sensorKey, int> dataArrived;    /**< incremental counter */
        int             triggerLine;
        bool            synched;        /**< tell if the sensor is synced with the log */
        int             width;
        int             height;
        //tk::data::DataType type;      /**< type of the sensor, used for visualization */

        /**
         * @brief Construct a new SensorInfo object
         */
        SensorInfo() 
        {
            name            = "???";
            version         = 1.0f;
            nSensors        = 1;
            synched         = false;
            //type            = tk::data::DataType::NOT_SPEC;
            triggerLine     = -1;
            width           = 0;
            height          = 0;
        }

        void operator=(SensorInfo i) noexcept {
            this->name              = i.name;
            this->version           = i.version;
            this->nSensors          = i.nSensors;
            this->dataArrived       = i.dataArrived;
            this->synched           = i.synched;
            //this->type              = i.type;
            this->triggerLine       = i.triggerLine;
            this->width             = i.width;
            this->height            = i.height;
        }
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
        virtual bool init(const YAML::Node conf, const std::string &name, LogManager *log = nullptr, const bool &aGui = true) final;

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
        SensorStatus status() const;

        /**
         * @brief   Get the TF object.
         * 
         * @param id tf id
         * @return tk::common::Tfpose 
         */
        tk::common::Tfpose getTf (const int id = 0) const;

        /**
         * @brief   Get vector of TF objects.
         * 
         * @return std::vector<tk::common::Tfpose> 
         */
        std::vector<tk::common::Tfpose> getTfs() const;

        /**
         * @brief   Get vector of types provided by the sensor
         * 
         * @return std::vector<tk::data::DataType> 
         */
        const std::vector<sensorKey>& getTypes() { return avaibleTypes; };

        

    protected:    
        SensorStatus                        senStatus;  /**< Sensor status */
        LogManager                          *log;       /**< Log Manager reference */
        std::map<sensorKey, SensorPool_t*>  pool;       /**< data pool */  
        int                                 poolSize;
        bool                                usingPool;
        std::vector<tk::common::Tfpose>     tf;         /**< Sensor TF */

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

        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        void addPool(int sensorID = 0)
        {
            tkASSERT(poolSize > 0);

            if (poolSize == 1)
                tkWRN("PoolSize is set to 1 element, this is not optimal even if you have just one reader.\n")

            tk::sensors::SensorPool_t *sPool = new tk::sensors::SensorPool_t;
            sPool->empty    = true;
            sPool->size     = poolSize;
            sPool->drw      = nullptr;
            sPool->lastStamps = std::vector<timeStamp_t>(4);
            sPool->pool.init<T>(sPool->size);

            // setup common header
            sPool->header.name       = info.name + "_" + tk::data::ToStr(T::type) + "_" + std::to_string(sensorID);
            sPool->header.type       = T::type;
            sPool->header.sensorID   = sensorID;
            sPool->header.tf         = getTf();

            int idx;
            for (int i = 0; i < sPool->size; i++) {
                auto data = dynamic_cast<T*>(sPool->pool.add(idx));
                
                // copy header to all pool data, name is update apart to skip WARN message in copy function
                data->header.name = sPool->header.name;
                data->header = sPool->header;

                sPool->pool.releaseAdd(idx);
            }
            
            // add pool
            pool.insert(std::make_pair(std::make_pair(T::type, sensorID), sPool));
            info.dataArrived.insert(std::make_pair(std::make_pair(T::type, sensorID), 0));
            avaibleTypes.push_back(std::make_pair(T::type, sensorID));
        }

    public:    
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        SensorPool_t* getPool(int sensorID = 0)
        {
            std::map<sensorKey, SensorPool_t*>::iterator it = pool.find(std::make_pair(T::type, sensorID)); 
            if (it == pool.end()) {
                tkERR("No pool present with this template type.\n");
                return nullptr;
            } else {
                return it->second;
            }
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
        void loop(sensorKey key);

    public:
        
        std::vector<sensorKey>              avaibleTypes; /**< data types provided by the sensor */ 

        /**
         * @brief   Extract last or newest element from the pool, based on timeout parameter value
         *          passed. Must be called after start() method.
         * 
         * @param data      returned data
         * @param idx       index of the data returned from the pool, used in release() method.
         * @param timeout   grab timeout [micro]
         * @return true     data is available
         * @return false    data is not available
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        bool grab(const T* &data, int &idx, uint64_t timeout = 0, int sensorID = 0) 
        {   
            // check if the passed template is the same as the pointer
            if (T::type != data->type) {
                tkERR("Type mismatch between template ("<<tk::data::ToStr(T::type)<<" and passed const pointer ("<<tk::data::ToStr(data->type)<<").\n");
                return false;
            }

            // check if template type is present inside pool
            std::map<sensorKey, SensorPool_t*>::iterator it = pool.find(std::make_pair(T::type, sensorID)); 
            if (it == pool.end()) {
                tkERR("No pool present with this template type "<<tk::data::ToStr(T::type)<<" and this ID "<<sensorID<<".\n");
                return false;
            }

            // check if pool is empty
            if (it->second->empty == true) {
                //tkWRN("Pool empty.\n");
                return false;
            }
            
            // grab
            if (timeout != 0) {     // locking
                data = dynamic_cast<const T*>(it->second->pool.get(idx, timeout));
            } else {                // non locking
                data = dynamic_cast<const T*>(it->second->pool.get(idx)); 
            }

            // error handling
            if (idx == -1) {
                tkWRN("Timeout.\n");
            } else if (idx == -2) {
                tkWRN("No free elemnt in the pool.\n");
            }

            if(data)
                postGrab(data);

            return (data != nullptr)?true:false;
        }

        virtual void postGrab(const tk::data::SensorData *data){}

        /**
         * @brief   Release grabbed element from the pool, must be called after a grab().
         * 
         * @tparam T  
         * @param idx   index of the data returned from the pool, given by the grab() method.
         */
        template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
        bool release(const int idx, int sensorID = 0)
        {
            // check if template type is present inside pool
            std::map<sensorKey, SensorPool_t*>::iterator it = pool.find(std::make_pair(T::type, sensorID)); 
            if (it == pool.end()) {
                tkERR("No pool present with this template type "<<tk::data::ToStr(T::type)<<" and this ID "<<sensorID<<".\n");
                return false;
            }

            // release element
            it->second->pool.releaseGet(idx);
            return true;
        }                
};
}}
