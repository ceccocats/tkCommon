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

            // starting from last,
            int i = 0;
            while(true) {
                // starting from last search for unlocked element
                id = (last + (size-i)) % size;
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

            id = freeID;
        gmtx.unlock();

        return dynamic_cast<tk::data::SensorData*>(data[id]);
    }

    /**
     * @brief   Method to release a pointer to sensor data in writing mode.
     * 
     * @param aID   index of the data inside of the pool
     * @param aSuccesfulRead 
     */
    void 
    releaseAdd(const int aID, bool aSuccesfulRead = false) {
        gmtx.lock();
            // check if was really locked
            if(!data[aID]->tryLock()) {
                locked--;
            }
            // unlock anyway
            data[aID]->unlockWrite(); 

            last = aID;

            if (aSuccesfulRead) {
                inserted++;
                data[aID]->header.messageID = inserted;
                
                // notify new data available
                cv.notify_all();
            }
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
        if (inserted == 0) {
            id = -1;
            return nullptr;
        }

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
     * @brief Get the data with the closest timestamp
     * 
     * @param id    index of the sensor data returned, used to release
     * @param stamp timestamp to search
     * @return const tk::data::SensorData* 
     */
    const tk::data::SensorData* 
    getStamped(int &id, timeStamp_t stamp) {
        
        if (inserted == 0) {
            id = -1;
            return nullptr;
        }

        gmtx.lock();

            // no free element in the pool
            if(locked >= size) {
                id  = -2;
                gmtx.unlock();
                return nullptr;
            }

            id = last;
            // starting from last,
            int a = last;
            int b;
            bool first = true;
            while(true) {
                // starting from last search for unlocked element
                b = a;
                if( !first || data[a]->tryLockRead() ){
                    int dt_a = std::abs(int(data[a]->header.stamp-stamp));
                    b = ( a + (size-1)) % size;
                    bool found = false;
                    while(!data[b]->tryLockRead()){
                        b = ( b + (size-1)) % size;
                    }
                    int dt_b = std::abs(int(data[b]->header.stamp-stamp));
                    if(dt_b >= dt_a){
                        // a is the closest
                        data[b]->unlockRead();
                        break;
                    }else{
                        // a is wrong
                        data[a]->unlockRead();
                        a = b;
                        first = false;
                    }
                }else{
                    a = (a + (size-1)) % size;
                }
            }

            id = a;
            if (data[id]->getReaders() == 1)
                locked++;
        gmtx.unlock();
        return dynamic_cast<const tk::data::SensorData*>(data[id]);
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