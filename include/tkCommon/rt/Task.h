#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/timerfd.h>

namespace tk { namespace rt {

class Task {

private:
	int timer_fd;

public:
    uint32_t period_us;
    uint64_t wakeups;
	uint64_t wakeups_missed;
    uint64_t last_ts;


     Task() {}
    ~Task() {}

    bool init(uint32_t period_us) {
        this->period_us = period_us;
        this->wakeups = 0;
        this->wakeups_missed = 0;


        int ret;
        unsigned int ns;
        unsigned int sec;
        int fd;
        struct itimerspec itval;

        /* Create the timer */
        fd = timerfd_create(CLOCK_MONOTONIC, 0);
        timer_fd = fd;
        if (fd == -1) {
            tkERR("could not init periodic task\n");
            return fd == 0;
        }

        /* Make the timer periodic */
        sec = period_us / 1000000;
        ns = (period_us - (sec * 1000000)) * 1000;
        itval.it_interval.tv_sec = sec;
        itval.it_interval.tv_nsec = ns;
        itval.it_value.tv_sec = sec;
        itval.it_value.tv_nsec = ns;
        ret = timerfd_settime(fd, 0, &itval, NULL);
        if(ret != 0) {
            tkERR("could not init periodic task\n");
        }
        return ret == 0;
    }

    void wait() {
        struct timespec tp;
        uint64_t missed;
        int ret;

        /* Wait for the next timer event. If we have missed any the
        number is written to "missed" */
        ret = read(timer_fd, &missed, sizeof(missed));
        if (ret == -1) {
            tkERR("wait error periodic task\n");
            perror("read timer");
            return;
        }
        clock_gettime(CLOCK_MONOTONIC, &tp);
        
        // update stats
        last_ts = uint64_t(tp.tv_sec)*1e6 + tp.tv_nsec/1000;
        wakeups++;
        wakeups_missed += (missed-1);
    }

    void printStats() {
        std::cout<<"TASK:\n";
        std::cout<<"\tperiod:  "<<period_us<<"\n";
        std::cout<<"\twakeups: "<<wakeups<<"\n";
        std::cout<<"\tmissed:  "<<wakeups_missed<<"\n";
    }

};

}}