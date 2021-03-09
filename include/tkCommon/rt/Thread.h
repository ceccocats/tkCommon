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
    pthread_t   th;

public:

     Thread() {}
    ~Thread() {}

    bool init(void *(*fun_ptr) (void *), void* args = nullptr) {
        return pthread_create(&th, NULL, fun_ptr, args) == 0;
    }

    void join() {
        pthread_join(th, NULL);
    }

};

}}