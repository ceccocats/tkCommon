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

        void lockWrite(){

            mutex.lock();
        }

        void unlockWrite(){

            counter++;
            modified = true;
            mutex.unlock();

        }

        void lockRead(){
            globalMutex.lock();
            
            if (nReader == 0){
                mutex.lock();
            }
            nReader++;
            globalMutex.unlock();
            
        }

        bool tryLockRead(){
            globalMutex.lock();

            bool l = true;
            
            if (nReader == 0){
                l = mutex.try_lock();
            }
            if( l )
                nReader++;
            globalMutex.unlock();

            return l;
        }

        void unlockRead(){
            globalMutex.lock();
            
            if (nReader == 1)
                mutex.unlock();
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
            return modified;
        }
};

}}