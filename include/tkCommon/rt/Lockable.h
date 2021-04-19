#pragma once
#include <mutex>
#include <condition_variable>
#include <tkCommon/rt/Profiler.h>

namespace tk { namespace rt {

class Lockable{
    private:
        mutable std::mutex  MTX;
        mutable std::mutex  gMTX;
        mutable std::mutex  cvMTX;

        mutable std::condition_variable CV;
        
        mutable uint32_t    counter;
        mutable uint32_t    nReader;
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
            if (MTX.try_lock()) {
                writing = true;
                return true;
            } else {
                return false;
            }
        }

        void lockRead() const {
            std::unique_lock<std::mutex> lck(cvMTX);
                while (writing) CV.wait(lck);
                gMTX.lock();
                if (nReader == 0)
                    MTX.lock();
                nReader++;
                gMTX.unlock();
            cvMTX.unlock();
        }

        void unlockRead() const {
            gMTX.lock();
                if (nReader == 1)
                    MTX.unlock();
                    
                if (nReader > 0)
                    nReader--; 
            gMTX.unlock();
        }


        bool tryLockRead() const {
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

        bool isChanged(uint32_t &counter) const {
            gMTX.lock();
                bool r = false;
                if (this->counter > counter) {
                    r       = true;
                    counter = this->counter;
                }
            gMTX.unlock();
            return r;
        }

        uint32_t getReaders() const {
            return nReader;
        }
};

}}