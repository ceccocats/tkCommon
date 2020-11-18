#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/GPSData.h"

#include <signal.h>
#include <pthread.h>

int size = 5;
tk::rt::DataPool            pool;
tk::data::GPSData           *data = nullptr;
const tk::data::GPSData     *cData = nullptr;
bool gRun = true;
std::mutex mtx;

void
my_handler(sig_atomic_t s)
{
    gRun = false;
    std::cout << "\nClosing...\n";
}

void
*writer(void*) 
{
    std::cout<<"Writer\n";

    int idx;
    int counter = 0;
    while(gRun) {
        data = dynamic_cast<tk::data::GPSData*>(pool.add(idx));

        data->lat = counter;
        data->lon = counter;

        mtx.lock();
        std::cout<<"Ho inserito "<<data->lat<<", "<<data->lon<<"\n";
        mtx.unlock();

        pool.releaseAdd(idx);

        counter++;
        usleep(10000);
    }

    pthread_exit(nullptr);
}

void
*reader(void*)
{
    std::cout<<"Reader "<<pthread_self()<<"\n";
    
    int idx;
    while(gRun) {
        cData = dynamic_cast<const tk::data::GPSData*>(pool.getNew(idx));
        if (cData!= nullptr)  {
            mtx.lock();
            std::cout<<"getNew "<<pthread_self()<<"\tlat "<<cData->lat<<", lon "<<cData->lon<<"\n";
            mtx.unlock();
            pool.releaseGet(idx);
        }
        //sleep(1);
    }

    pthread_exit(nullptr);
}

void
*reader2(void*)
{
    std::cout<<"Reader "<<pthread_self()<<"\n";
    
    int idx;
    while(gRun) {
        cData = dynamic_cast<const tk::data::GPSData*>(pool.getLast(idx));
        
        mtx.lock();
        std::cout<<"getLast"<<"\tlat "<<cData->lat<<", lon "<<cData->lon<<"\n";
        mtx.unlock();
        pool.releaseGet(idx);

        usleep(5000);
    }

    pthread_exit(nullptr);
}


int main( int argc, char** argv){
    signal(SIGINT, my_handler);
    int idx;
    // init pool data 
    pool.init<tk::data::GPSData>(10);

    // spawn threads
    pthread_t pt1, pt2, pt3, pt4;
	pthread_create(&pt1, nullptr, writer, nullptr);
    pthread_create(&pt2, nullptr, reader, nullptr);
    pthread_create(&pt3, nullptr, reader, nullptr);
    pthread_create(&pt4, nullptr, reader2, nullptr);

    pthread_join(pt1, nullptr);
    pthread_join(pt2, nullptr);
    pthread_join(pt3, nullptr);
    pthread_join(pt4, nullptr);

    return 0;
}