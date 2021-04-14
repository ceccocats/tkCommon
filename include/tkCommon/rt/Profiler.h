#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <tkCommon/time.h>
#include <tkCommon/exceptions.h>

namespace tk { namespace rt {

class Profiler {

private:
    static Profiler *instance;

public:
    struct ProfileInfo_t {
        timeStamp_t startTs = 0;
        uint64_t count = 0;
        uint64_t sum = 0;

        int avg = 0;
        int min = 0, max = 0;

        void print() {
            std::cout<<"Avg: "<<avg<<" Min: "<<min<<" Max: "<< max<<" (us)\n";
        }
    };
    std::map<std::string, ProfileInfo_t> infos;

     Profiler() {
    }
    ~Profiler() {}

    static Profiler* getInstance(){
        static Profiler inst;
        return &inst;
    }

    void tic(std::string key) {
        tic(&infos[key]);
    }
    void toc(std::string key) {
        toc(&infos[key]);
    }

    void tic(ProfileInfo_t *pi) {
        pi->startTs = getTimeStamp();          
    }

    void toc(ProfileInfo_t *pi) {
        tkASSERT(pi->startTs != 0, "must tic() before toc()")
        int diff = getTimeStamp() - pi->startTs; 
        
        pi->count++;
        pi->sum += diff;
        if(pi->max == 0 || pi->max < diff) pi->max = diff;
        if(pi->min == 0 || pi->min > diff) pi->min = diff;
        pi->avg = pi->sum / pi->count;
    }

    void print() {
        std::cout<<"Profiler:\n";
        for (auto& i : infos) {
            std::cout<<i.first<<"   ";
            i.second.print();
        }
    }
};


}}

#define tkPROF_tic(name) tk::rt::Profiler::getInstance()->tic(#name);
#define tkPROF_toc(name) tk::rt::Profiler::getInstance()->toc(#name);
#define tkPROF_print     tk::rt::Profiler::getInstance()->print();