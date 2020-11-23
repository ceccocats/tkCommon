#pragma once
#include <mutex>
#include <tkCommon/rt/Profiler.h>

namespace tk { namespace rt {

class Lockable{
    private:
        std::mutex  mutex;
        std::mutex  globalMutex;
        uint32_t    counter;
        uint32_t    nReader;
        bool        modified;
        bool        writing;

    public:
        Lockable() {
            globalMutex.unlock();
            mutex.unlock();

            counter     = 0;
            nReader     = 0;
            modified    = false;
            writing     = false;
        }

        void lockWrite(){
            tkPROF_tic(lock)
            writing = true;
            mutex.lock();
            writing = false;
            tkPROF_toc(lock)
        }

        void unlockWrite(){

            counter++;
            modified = true;
            mutex.unlock();

        }

        void lockRead(){
            while(writing){
                usleep(1);
            }
            globalMutex.lock();
            
            if (nReader == 0){
                mutex.lock();
            }
            nReader++;
            globalMutex.unlock();
            
        }

        bool tryLockRead(){
            if(writing) return false;

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
            bool l = false;
            if(mutex.try_lock()){

                l = modified;
                modified = false;
                mutex.unlock();
            }
            return l;
        }
};

}}