#pragma once
#include <mutex>

namespace tk { namespace rt {

class Lockable{
    private:
        std::mutex  mutex;
        std::mutex  globalMutex;
        uint32_t    counter;
        uint32_t    nReader;
        bool        modified;

    public:
        Lockable() {
            globalMutex.unlock();
            mutex.unlock();

            counter     = 0;
            nReader     = 0;
            modified    = false;
        }

        bool lockWrite(){
            globalMutex.lock();
            if (nReader == 0 && mutex.try_lock()) {
                nReader++;
                globalMutex.unlock();
                return true;
            } else {
                globalMutex.unlock();
                return false;
            }
        }

        void unlockWrite(){
            globalMutex.lock();
                counter++;
                nReader--;
                modified = true;
                mutex.unlock();
            globalMutex.unlock();
        }

        bool lockRead(){
            globalMutex.lock();
            if (mutex.try_lock()) {
                mutex.unlock();
                nReader++;
                
                globalMutex.unlock();
                return true;
            } else {
                globalMutex.unlock();
                return false;
            }
        }

        void unlockRead(){
            globalMutex.lock();
                nReader--;
            globalMutex.unlock();
        }

        bool tryLock() {
            return mutex.try_lock();
        }

        uint32_t getModCount(){
            return counter;
        }

        bool isChanged(){
            globalMutex.lock();
            if(mutex.try_lock() == true){
                bool status = modified;
                modified = false;
                mutex.unlock();
                globalMutex.unlock();
                return status;
            } else {
                globalMutex.unlock();
                return false;
            }
        }
};

}}