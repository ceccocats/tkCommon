#pragma once

#include <mutex>             
#include <condition_variable>
#include <chrono>
#include <vector>
#include "tkCommon/data/SensorData.h"

namespace tk { namespace rt {
/**
 * 
 */
class DataPool {
private:

    std::mutex  gmtx;
    std::condition_variable cv; 

    tk::data::SensorData **data;
    int         *nReader;
   
    int         last;
    int         locked;
    int         size;
    bool        initted;
public:
    /**
     * @brief Construct a new DataPool object
     * 
     */
    DataPool() {   
        last        = 0;
        locked      = 0;
        size        = 0;
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
            nReader = new int[size];
            for (int i = 0; i < size; i++) {
                data[i] = new T();
                data[i]->init();
                data[i]->unlockRead();
                nReader[i] = 0;
            }           
            initted = true;
        }
        gmtx.unlock();
    }

    /**
     * @brief 
     * 
     * @param id 
     * @return tk::data::SensorData* 
     */
    tk::data::SensorData* 
    add(int &id) {
        gmtx.lock();
            // no space in the pool
            if(locked >= size) {
                id  = -1;
                gmtx.unlock();
                return nullptr;
            }

            // search first free object        
            int freeID = (last + 1) % size;
            while(true) {
                if(nReader[freeID] == 0 && data[freeID]->tryLock()) {
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
     * @brief 
     * 
     * @param id 
     */
    void 
    releaseAdd(int id) {
        gmtx.lock();

        // check if was really locked
        if(!data[id]->tryLock()) {
            locked--;
        }
        
        // unlock anyway
        data[id]->unlockWrite(); 

        // notify new data available
        cv.notify_all();

        gmtx.unlock();
    }

    /**
     * @brief Get the Last object
     * 
     * @param id 
     * @return const tk::data::SensorData* 
     */
    const tk::data::SensorData* 
    getLast(int &id) {
        gmtx.lock();

            id = last;
            // starting from last,
            int i = 0;
            while(true) {
                // starting from last search for unlocked element
                id = (id + (size-i)) % size;
                if(data[id]->tryLock()) {
                    nReader[id]++;
                    data[id]->unlockRead();
                    break;
                }
                i++;
            }
        gmtx.unlock();

        return dynamic_cast<const tk::data::SensorData*>(data[id]);
    }

    /**
     * @brief Get the New object
     * 
     * @param id 
     * @return const tk::data::SensorData* 
     */
    const tk::data::SensorData*
    getNew(int &id) {
        std::unique_lock<std::mutex> lck(gmtx);
            if (cv.wait_for(lck, std::chrono::seconds(1)) == std::cv_status::timeout) {
                id = -1;
                lck.unlock();
                return nullptr;
            }
        lck.unlock();
        return getLast(id);
    }

    /**
     * @brief 
     * 
     * @param id 
     */
    void 
    releaseGet(int id) {
        gmtx.lock();
            nReader[id]--;
        gmtx.unlock();
    }

    /**
     * @brief 
     * 
     */
    void close() {
        gmtx.lock();
        if (initted) {
            delete[] nReader;
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