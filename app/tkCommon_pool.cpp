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

void pausetta(void) {
  struct timespec t;
  t.tv_sec = 0;
  t.tv_nsec = (rand()%10+1)*1000000;
  nanosleep(&t,NULL);
}

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
        
        if (data != nullptr) {
            data->gps.lat = counter;
            data->gps.lon = counter;

            //mtx.lock();
            printf("write\tlat %f, lon %f\n", data->gps.lat, data->gps.lon);
            //mtx.unlock();

            pool.releaseAdd(idx);

            counter++;
        }
        pausetta();
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
        if (cData != nullptr)  {
            //mtx.lock();
            printf("new\tlat %f, lon %f\n", cData->gps.lat, cData->gps.lon);
            //mtx.unlock();
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
        if (cData != nullptr) {
            //mtx.lock();
            printf("last\tlat %f, lon %f\n", cData->gps.lat, cData->gps.lon);
            //mtx.unlock();
            pool.releaseGet(idx);
        }
        pausetta();
    }

    pthread_exit(nullptr);
}


int main( int argc, char** argv){
    signal(SIGINT, my_handler);
    int idx;
    // init pool data 
    pool.init<tk::data::GpsImuData>(10);

    // spawn threads
    pthread_t pt[50];
    
	pthread_create(&pt[0], nullptr, writer, nullptr);
    for (int i = 1; i < 20; i++) 
        pthread_create(&pt[i], nullptr, reader, nullptr);
    for (int i = 20; i < 25; i++)
        pthread_create(&pt[i], nullptr, reader2, nullptr);

    for (int i = 0; i < 25; i++)
    pthread_join(pt[i], nullptr);

    return 0;
}