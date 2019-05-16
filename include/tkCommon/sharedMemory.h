#include <iostream> 
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <stdio.h> 

namespace tk{ namespace common{

template <class T>
class sharedMemory{

    private:
        T*  data;
        int type;
        int shmid;
    public:

        bool init(std::string name, int id ){

            key_t key = ftok(name.c_str(),id);
            shmid = shmget(key,1024,0666|IPC_CREAT);
            data = (T*) shmat(shmid,(void*)0,0);
        }

        T read(){

            return *data;
        }

        bool write(T d){

            data = *d;
        }

        void close(){

            shmctl(shmid,IPC_RMID,NULL);
        }



};

}}