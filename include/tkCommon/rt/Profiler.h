#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <map>
#include <thread>
#include <tkCommon/time.h>
#include <tkCommon/exceptions.h>

namespace tk { namespace rt {

class Profiler {

private:
    static Profiler *instance;
    std::mutex m;

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
    std::map<std::string, std::map<std::thread::id,ProfileInfo_t>> infos;

     Profiler() {
    }
    ~Profiler() {}

    static Profiler* getInstance(){
        static Profiler inst;
        return &inst;
    }

    void tic(const std::string &key) {
        std::thread::id this_id = std::this_thread::get_id();
        m.lock();
        tic(&infos[key][this_id]);
        m.unlock();
    }
    void toc(const std::string &key) {
        std::thread::id this_id = std::this_thread::get_id();
        m.lock();
        toc(&infos[key][this_id]);
        m.unlock();
    }

private:
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

public:
    void print() {
        std::cout<<"Profiler:\n";
        for (auto& i : infos) {
            ProfileInfo_t p;
            bool first = true;
            for(auto&j : i.second) {
                p.sum += j.second.sum;
                p.count += j.second.count;
                if(first) {
                    p.min = j.second.min;
                    p.max = j.second.max;
                    first = false;
                }
                if(j.second.min < p.min)
                    p.min = j.second.min;
                if(j.second.max > p.max)
                    p.max = j.second.max;
            }
            p.avg = p.sum / p.count;

            std::cout<<i.first<<"   ";
            p.print();
        }
    }
};


}}

#ifndef tkPROF_disable
    #define tkPROF_tic(name) tk::rt::Profiler::getInstance()->tic(#name);
    #define tkPROF_toc(name) tk::rt::Profiler::getInstance()->toc(#name);
    #define tkPROF_print     tk::rt::Profiler::getInstance()->print();
#endif
