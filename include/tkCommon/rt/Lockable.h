#pragma once
#include <mutex>
#include <condition_variable>
#include <tkCommon/rt/Profiler.h>

namespace tk { namespace rt {

class Lockable{
    private:
        std::mutex  MTX;
        std::mutex  gMTX;
        std::mutex  cvMTX;

        std::condition_variable CV;
        
        uint32_t    counter;
        uint32_t    nReader;
        bool        writing;
        
    public:
        Lockable() {
            gMTX.unlock();
            MTX.unlock();

            counter     = 0;
            nReader     = 0;
            writing     = false;
        }

        void lockWrite() {
            writing = true;
            MTX.lock();
        }

        void unlockWrite() {
            gMTX.lock();
                if (writing) {
                    counter++;
                    writing     = false;
                }
                MTX.unlock();
                CV.notify_all();
            gMTX.unlock();
        }

        bool tryLock() {
            return MTX.try_lock();
        }

        void lockRead() {
            std::unique_lock<std::mutex> lck(cvMTX);
                while (writing) CV.wait(lck);
                gMTX.lock();
                if (nReader == 0)
                    MTX.lock();
                nReader++;
                gMTX.unlock();
            cvMTX.unlock();
        }

        void unlockRead() {
            gMTX.lock();
                if (nReader == 1)
                    MTX.unlock();
                    
                if (nReader > 0)
                    nReader--; 
            gMTX.unlock();
        }


        bool tryLockRead() {
            gMTX.lock();
                bool r = false;
                if (nReader > 0)
                    r = true;
                else 
                    r = MTX.try_lock();
                
                if (r == true)
                    nReader++;
            gMTX.unlock();
            return r;
        }

        bool isChanged(uint32_t &counter) {
            gMTX.lock();
                bool r = false;
                if (this->counter > counter) {
                    r       = true;
                    counter = this->counter;
                }
            gMTX.unlock();
            return r;
        }
};

}}