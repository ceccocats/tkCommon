#include <iostream>
#include <boost/interprocess/managed_shared_memory.hpp>

using namespace boost::interprocess;

namespace tk{ namespace common{

template <class T>
class sharedMemory{

    private:
        std::string                     name;

        std::pair<T*, std::size_t>      ref;

    public:

        bool init(std::string name){

            this->name = name;

            try{
                managed_shared_memory segment(open_only, this->name.c_str());
                ref = segment.find<T>(this->name.c_str());
            }catch(...){
                managed_shared_memory segment(create_only, this->name.c_str(), 200);
                T *r = segment.construct<T>(this->name.c_str())();
                ref = segment.find<T>(this->name.c_str());
            }
        }

        T read(){

            return *ref.first;
        }

        bool write(T data){

            ref.first = data;
            return true;
        }

        void close(){

            shared_memory_object::remove(this->name.c_str());
        }



};

}}