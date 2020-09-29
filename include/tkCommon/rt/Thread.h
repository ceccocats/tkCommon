#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include "tkCommon/terminalFormat.h"

namespace tk { namespace rt {

class Thread {

private:
    pthread_t th;

public:

     Thread() {}
    ~Thread() {}

    bool init(void *(*fun_ptr) (void *)) {
        int err = pthread_create(&th, NULL, fun_ptr, NULL);
        return err == 0;
    }

    void join() {
        pthread_join(th, NULL);
    }

};

}}