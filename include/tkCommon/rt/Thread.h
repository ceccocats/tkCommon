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

public:

     Thread() { started = false; }
    ~Thread() {}

    bool init(void *(*fun_ptr) (void *), void* args = nullptr) {
        if(pthread_create(&th, NULL, fun_ptr, args) == 0) {
            started = true;
            return true;
        }
        return false;
    }

    void join() {
        if(started)
            pthread_join(th, NULL);
    }

};

}}