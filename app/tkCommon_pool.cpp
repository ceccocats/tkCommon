#include "tkCommon/rt/Pool.h"
#include "tkCommon/data/GpsImuData.h"

#include <signal.h>
#include <pthread.h>

int size = 5;
tk::rt::DataPool            pool;
tk::data::GpsImuData           *data = nullptr;
const tk::data::GpsImuData     *cData = nullptr;
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
        data = dynamic_cast<tk::data::GpsImuData*>(pool.add(idx));

        data->gps.lat = counter;
        data->gps.lon = counter;

        mtx.lock();
        std::cout<<"Ho inserito "<<data->gps.lat<<", "<<data->gps.lon<<"\n";
        mtx.unlock();

        pool.releaseAdd(idx);

        counter++;
        usleep(1000);
    }

    pthread_exit(nullptr);
}

void
*reader(void*)
{
    std::cout<<"Reader "<<pthread_self()<<"\n";

    int idx;
    while(gRun) {
        cData = dynamic_cast<const tk::data::GpsImuData*>(pool.get(idx, (uint64_t)1e4));
        if (cData!= nullptr)  {
            mtx.lock();
            std::cout<<"new\t"<<pthread_self()<<"\tlat "<<cData->gps.lat<<", lon "<<cData->gps.lon<<"\n";
            mtx.unlock();
            pool.releaseGet(idx);
        }
    }

    pthread_exit(nullptr);
}

void
*reader2(void*)
{
    std::cout<<"Reader "<<pthread_self()<<"\n";
    
    int idx;
    while(gRun) {
        cData = dynamic_cast<const tk::data::GpsImuData*>(pool.get(idx));
        
        mtx.lock();
        std::cout<<"last\t"<<"\tlat "<<cData->gps.lat<<", lon "<<cData->gps.lon<<"\n";
        mtx.unlock();
        pool.releaseGet(idx);

        usleep(500);
    }

    pthread_exit(nullptr);
}


int main( int argc, char** argv){
    signal(SIGINT, my_handler);
    int idx;
    // init pool data 
    pool.init<tk::data::GpsImuData>(10);

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