#pragma once
#include <mutex>

namespace tk { namespace rt {

class Lockable{
    private:
        std::mutex  mutex;
        uint32_t    counter;
        bool        modified = false;

    public:
        void lock(){
            mutex.lock();
        }

        void unlockWrite(){
            counter++;
            modified = true;
            mutex.unlock();
        }

        void unlockRead(){
            mutex.unlock();
        }

        bool tryLock() {
            return mutex.try_lock();
        }

        uint32_t getModCount(){
            return counter;
        }

        bool isChanged(){
            if(mutex.try_lock() == true){
                bool status = modified;
                modified = false;
                mutex.unlock();
                return status;
            }else{
                return false;
            }
        }
};

}}