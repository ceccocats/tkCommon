#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include "tkCommon/common.h"

namespace tk { namespace rt {


class Thread {

private:
    bool started;
    pthread_t   th;

    // settings
    std::string name;
    int prio;

public:

     Thread(std::string aName = "", int aPrio = 20) { 
        started = false; 
        prio = aPrio;
        name = aName;
    }
    ~Thread() {}


    bool init(void *(*fun_ptr) (void *), void* args = nullptr) {

        pthread_attr_t      attr;
        pthread_attr_init(&attr);
        //if(pthread_attr_setschedpolicy(&attr, SCHED_FIFO) != 0) {
        //    perror("pthread_attr_setschedpolicy");
        //    tkFATAL("could not set sched policy");            
        //} 
        //if(pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED) != 0){
        //    perror("pthread_attr_setinheritsched");
        //    tkFATAL("could not set param");
        //}

        // set prio
        //sched_param spar;
        //spar.sched_priority = prio;
        //if(pthread_attr_setschedparam(&attr, &spar) != 0){
        //    perror("pthread_attr_setschedparam");
        //    tkFATAL("could not set param");
        //}

        if(pthread_create(&th, &attr, fun_ptr, args) != 0) {
            perror("pthread_create");
            tkFATAL("could not create thread");
        }

        if(name != "") {
            if(!pthread_setname_np(th, name.c_str()) != 0) {
                perror("pthread_setname_np");
                tkFATAL("could not set name thread");
            }

        }
        started = true;
        return true;
    }

    void join() {
        if(started)
            pthread_join(th, NULL);
    }

    void printInfo() {
        struct sched_param param;
        int policy;
        pthread_getschedparam(th, &policy, &param);
        printf("Priority of the thread: %d, current policy is: %d\n",
                param.sched_priority, policy);
    }
};

}}