#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <sys/time.h>
#include "yaml-cpp/yaml.h" 

typedef uint64_t timeStamp_t;
inline timeStamp_t getTimeStamp() {
    struct timeval cur_time;
    gettimeofday(&cur_time, NULL);
    return timeStamp_t(cur_time.tv_sec)*1e6 + cur_time.tv_usec;
}
inline timeStamp_t tv2TimeStamp(struct timeval tv) {
    return timeStamp_t(tv.tv_sec)*1e6 + tv.tv_usec;
} 

template<typename T>
T clamp(T val, T min, T max) {
    if(val < min)
        return min;
    if(val > max)
        return max;
    return val;
}

template<typename T>
T getYAMLconf(YAML::Node conf, std::string key, T defaultVal) {
    T val = defaultVal;
    if(conf && conf[key]) {
        val = conf[key].as<T>();
    }
    //std::cout<<"YAML "<<key<<", val: "<<val<<"\n";
    return val;
}

/* 
    Class for loop rate at a certain delta time
    in microsecs
*/
struct LoopRate {

    std::string name;
    timeStamp_t dt;
    timeStamp_t T, lastT;

    LoopRate(timeStamp_t _dt, std::string _name = "anonymous") {
        dt  = _dt;
        name = _name;
        T = lastT = 0;
    }

    void wait() {
        T = getTimeStamp();

        timeStamp_t delta = T - lastT;
        if(delta >= 0 && delta < dt) {
            usleep(dt - delta);
        } else if(lastT > 0) {
            std::cout<<"LOOPRATE "<<name<<" exceeded: "<<delta<<" of "<<dt<<" ms\n";
        }

        lastT = getTimeStamp();
    }
};

template<typename T>
struct CircularArray {

    static const int MAX_DIM = 1000;
    T array[CircularArray::MAX_DIM];
    int dim, position;

    std::mutex m;

    CircularArray(int _dim = 100) {
        setDim(_dim);
        position = 0;
    }

    void setDim(int _dim) {
        if(dim > CircularArray::MAX_DIM)
            dim = CircularArray::MAX_DIM;
        dim = _dim;
    }

    void add(T element) {
        std::lock_guard<std::mutex> lock(m);

        array[position % dim] = element;
        position++;
    }

    T head(int n) {
        std::lock_guard<std::mutex> lock(m);
        
        int index = position - 1 - n;
        index = (index + dim) % dim;
        return array[index];
    }

    T max() {
        T val = -9999999999;
        int idx = 0;
        for(int i=0; i<dim; i++) {
            if(array[i] > val) {
                val = array[i];
                idx = i;
            }
        }
        return array[idx];
    }

    T min() {
        T val = +9999999999;
        int idx = 0;
        for(int i=0; i<dim; i++) {
            if(array[i] < val) {
                val = array[i];
                idx = i;
            }
        }
        return array[idx];
    }

    int size() {
        if(position < dim)
            return position;
        else
            return dim;
    }

    void dump(std::string file) {     
        std::ofstream os;
        os.open(file);
        for(int i=0; i<size(); i++) {
            os<<head(i)<<"\n";
        }
        os.close();
    }
};

template<typename T>
struct FIFOQueue {

    static const int MAX_DIM = 1000;
    T array[FIFOQueue::MAX_DIM];
    int dim, inserted, position;

    std::mutex m;

    FIFOQueue(int _dim=1) {
        setDim(_dim);
        position = 0;
        inserted = 0;
    }
    
    void setDim(int _dim) {
        if(dim > FIFOQueue::MAX_DIM)
            dim = FIFOQueue::MAX_DIM;
        dim = _dim;
    }

    void add(T element) {
        std::lock_guard<std::mutex> lock(m);

        array[(position + inserted) % dim] = element;
        inserted++;
        if(inserted > dim)
            inserted = dim;
    }

    bool get(T &out) {  
        std::lock_guard<std::mutex> lock(m);
        
        if(inserted > 0) {
            out = array[position % dim];
            position++;
            inserted--;
            return true;
        } else {
            return false;
        }
    }
};


template<typename T>
struct PoolQueue {

    static const int MAX_DIM = 64;
    T array[PoolQueue::MAX_DIM];
    int dim;

    std::mutex gmtx; 
    std::mutex mtx[PoolQueue::MAX_DIM];
    int        last = 0;
    int        locked = 0;
    int        inserted = 0;

    PoolQueue(int _dim=1) {
        setDim(_dim);
        for(int i=0; i<MAX_DIM; i++) {
            mtx[i].unlock();
        }
    }
    
    void setDim(int _dim) {
        if(_dim > PoolQueue::MAX_DIM)
            _dim = PoolQueue::MAX_DIM;
        dim = _dim;
    }

    int add(T element) {
        
        gmtx.lock();
            // no space in the pool
            if(locked >= dim) {
                gmtx.unlock();
                return -1;
            }

            // search first free object        
            int insert_to = (last + 1) % dim;
            while(true) {
                if(mtx[ insert_to%dim ].try_lock())
                    break;
                insert_to++;
            }
            inserted++;
        gmtx.unlock();

        array[(insert_to) % dim] = element;

        gmtx.lock();
            last = insert_to;
        gmtx.unlock();
        return insert_to;
    }

    int getIdx(/*int idx,*/ T &out) {    

        gmtx.lock();
        int idx = last;
        if(mtx[ idx%dim ].try_lock()) {
            out = array[ idx%dim ];
            locked++;
        } else {
            idx = -1;
        }    
        gmtx.unlock();
 

        return idx;
    }

    void releaseIdx(int idx) {
        gmtx.lock();

        // check if was really locked
        if(!mtx[ idx%dim ].try_lock()) {
            locked--;
        }
        mtx[ idx%dim ].unlock();  // unlock anyway
        gmtx.unlock();          
    }

};
