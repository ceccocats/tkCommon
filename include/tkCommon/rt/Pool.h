#pragma once

#include <mutex>             
#include <condition_variable>
#include <chrono>
#include <vector>
#include "tkCommon/data/SensorData.h"

namespace tk { namespace rt {
/**
 * @brief Class to handle a pool of sensor data.
 * 
 */
class DataPool {
private:

    std::mutex  gmtx;
    std::condition_variable cv; 

    tk::data::SensorData **data;
   
    int         last;
    int         locked;
    int         size;
    bool        initted;

    /**
     * @brief Get the Last object
     * 
     * @param id    index of the sensor data returned, used to release
     * @return const tk::data::SensorData* 
     */
    const tk::data::SensorData* 
    getLast(int &id) {
        gmtx.lock();

            // no free element in the pool
            if(locked >= size) {
                id  = -2;
                gmtx.unlock();
                return nullptr;
            }

            id = last;
            // starting from last,
            int i = 0;
            while(true) {
                // starting from last search for unlocked element
                id = (id + (size-i)) % size;
                if(data[id]->tryLockRead())
                    break;
                i++;
            }

            if (data[id]->getReaders() == 1)
                locked++;
        gmtx.unlock();
        return dynamic_cast<const tk::data::SensorData*>(data[id]);
    }

public:
    int inserted;
    
    /**
     * @brief Construct a new DataPool object
     * 
     */
    DataPool() {   
        last        = 0;
        locked      = 0;
        size        = 0;
        inserted    = 0;
        initted     = false;
        gmtx.unlock();
    }

    /**
     * @brief Destroy the DataPool object
     * 
     */
    ~DataPool() {
        if (initted)
            close();
    }

    /**
     * @brief 
     * 
     * @param dim 
     */
    template<typename T, typename = std::enable_if<std::is_base_of<tk::data::SensorData, T>::value>>
    void 
    init(int dim) {
        gmtx.lock();
        if (!initted) {   
            size    = dim;
            data    = new tk::data::SensorData*[dim];
            for (int i = 0; i < size; i++) {
                data[i] = new T();
                data[i]->init();
            }           
            initted = true;
        }
        gmtx.unlock();
    }

    /**
     * @brief Method to request a pointer to sensor data in writing mode.
     * 
     * @param id    index of the sensor data returned, used to release
     * @return tk::data::SensorData* 
     */
    tk::data::SensorData* 
    add(int &id) {
        gmtx.lock();
            // no free element in the pool
            if(locked >= size) {
                id  = -2;
                gmtx.unlock();
                return nullptr;
            }

            // search first free object        
            int freeID = (last + 1) % size;
            while(true) {
                if(data[freeID]->tryLock()) {
                    locked++;
                    break;
                }
                    
                freeID = (freeID+1) % size;
            }

            last    = freeID;
            id      = freeID;
        gmtx.unlock();

        return dynamic_cast<tk::data::SensorData*>(data[last]);
    }

    /**
     * @brief Method to release a pointer to sensor data in writing mode.
     * 
     * @param id   index of the data inside of the pool
     */
    void 
    releaseAdd(const int id) {
        gmtx.lock();
            // check if was really locked
            if(!data[id]->tryLock()) {
                locked--;
            }
            
            // unlock anyway
            data[id]->unlockWrite(); 

            inserted++;

            // notify new data available
            cv.notify_all();
        gmtx.unlock();
    }

    /**
     * @brief Method to request a pointer to sensor data in reading mode.
     * 
     * @param id        index of the sensor data returned, used to release
     * @param timeout   optional timeout, if passed the get will return the newest data when available
     * @return const tk::data::SensorData* 
     */
    const tk::data::SensorData* 
    get(int &id, uint64_t timeout = 0) {
        if (timeout != 0) {
            std::unique_lock<std::mutex> lck(gmtx);
            if (cv.wait_for(lck, std::chrono::microseconds(timeout)) == std::cv_status::timeout) { // locking get
                id = -1;
                lck.unlock();
                return nullptr;
            } else {
                lck.unlock();
                return getLast(id);    
            }
        } else {
            return getLast(id);
        } 
    }


    
    /**
     * @brief Method to release a pointer to sensor data in reading mode.
     * 
     * @param id 
     */
    void 
    releaseGet(const int id) {
        gmtx.lock();
            data[id]->unlockRead();

            if (data[id]->getReaders() == 0)
                locked--;
        gmtx.unlock();
    }

    /**
     * @brief Method 
     * 
     * @param id 
     * @return true     if the id passed is lower than the internal one
     * @return false 
     */
    bool
    newData(const int id) {
        return (inserted > id)?true:false;
    }

    /**
     * @brief Method to deallocate memory.
     * 
     */
    void close() {
        gmtx.lock();
        if (initted) {
            for (int i = 0; i < size; i++){
                delete data[i];
            } 
            delete[] data;
            initted = false;
        }
        gmtx.unlock();
    }
};

}}