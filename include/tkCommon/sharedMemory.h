#include <iostream> 
#include <sys/ipc.h> 
#include <sys/shm.h> 
#include <stdio.h>
#include <string.h> 

namespace tk{ namespace common{

template <class T>
class sharedMemory{

    private:
        char*  data;
        int type;
        int shmid;
    public:

        bool init(std::string name, int id ){

            key_t key = ftok(name.c_str(),id);
            shmid = shmget(key,1024,0666|IPC_CREAT);
            data = (char*) shmat(shmid,(void*)0,0);
        }

        T read(){
		T a;
		memcpy(&a,data,sizeof(T));
            return a;
        }

        bool write(T d){
		memcpy(data,&d,sizeof(T));
        }

        void close(){

            shmctl(shmid,IPC_RMID,NULL);
        }



};

}}
