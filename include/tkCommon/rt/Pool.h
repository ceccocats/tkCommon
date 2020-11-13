#pragma once

#include <mutex>             
#include <condition_variable>
#include <chrono>
//#include <semaphore.h> 
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace common {
/**
 * 
 */
template<typename T, typename = std::enable_if<std::is_base_of<tk::rt::Lockable, T>::value>>
class Pool {
private:
    T           *data;
    std::mutex  gmtx;
    //std::mutex  cvMtx;
    int         *nReader;
    std::condition_variable cv;
    //sem_t       *sem; 

    int         last;
    int         locked;
    int         size;
    bool        initted;
    bool        newData;
public:
    Pool() {   
        last        = 0;
        locked      = 0;
        size        = 0;
        newData     = false;
        initted     = false;
        gmtx.unlock();
    }

    ~Pool() {
        if (initted)
            close();
    }

    void init(int dim) {
        gmtx.lock();
        if (!initted) {   
            size  = dim;
            data  = new T[size];
            nReader = new int[size];
            //sem   = new sem_t[size];
            for (int i = 0; i < size; i++) {
                data[i].unlockRead();
                nReader[i] = 0;
                //sem_init(sem[i], 0, 0);
            }           
            initted = true;
        }
        gmtx.unlock();
    }

    T* add(int &id) {
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
                if(nReader[freeID] == 0 && data[freeID].tryLock()) {
                    locked++;
                    break;
                }
                    
                freeID = (freeID+1) % size;
            }

            last    = freeID;
            id      = freeID;
        gmtx.unlock();

        return &data[last];
    }

    void releaseAdd(int id) {
        gmtx.lock();

        // check if was really locked
        if(!data[id].tryLock()) {
            locked--;
        }
        
        // unlock anyway
        data[id].unlockWrite(); 

        // notify new data available
        cv.notify_all();

        gmtx.unlock();
    }

    const T* getLast(int &id) {
        gmtx.lock();

        id = last;
        // starting from last,
        int i = 0;
        while(true) {
            // starting from last search for unlocked element
            id = (id + (size-i)) % size;
            if(data[id].tryLock()) {
                nReader[id]++;
                data[id].unlockRead();
                break;
            }
            i++;
        }
        gmtx.unlock();

        return &data[id];
    }

    const T* getNew(int &id) {
        std::unique_lock<std::mutex> lck(gmtx);
        if (cv.wait_for(lck, std::chrono::milliseconds(100)) == std::cv_status::timeout) {
            id = -1;
            lck.unlock();
            return nullptr;
        }
        lck.unlock();
        return getLast(id);
    }

    void releaseGet(int id) {
        gmtx.lock();
        nReader[id]--;
        gmtx.unlock();
    }

    void close() {
        gmtx.lock();
        if (initted) {
            
            initted = false;
        }
        gmtx.unlock();
    }
};

}}