#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <sys/time.h>
#include <unistd.h>
#include "yaml-cpp/yaml.h" 

/**
 * Timestamp value
 * espessed in microseconds from epoch
 */
typedef uint64_t timeStamp_t;

/**
 * get current timestamp
 * @return microseconds from epoch
 */
inline timeStamp_t getTimeStamp() {
    struct timeval cur_time;
    gettimeofday(&cur_time, NULL);
    return timeStamp_t(cur_time.tv_sec)*1e6 + cur_time.tv_usec;
}

/**
 * Convert timeval to microseconds from epoch
 * @param tv
 * @return
 */
inline timeStamp_t tv2TimeStamp(struct timeval tv) {
    return timeStamp_t(tv.tv_sec)*1e6 + tv.tv_usec;
} 

inline std::string getTimeStampString(std::string sep="--_--") {

    if(sep.length() != std::string("--_--").length())
        sep = "--_--";

    // current date/time based on current system
    time_t now = time(0);

    // convert now to string form
    char* dt = ctime(&now);

    // convert now to tm struct for UTC
    tm *gmtm = gmtime(&now);

    std::string datetime = std::to_string(gmtm->tm_year+1900);
    datetime += sep[0];
    datetime += std::to_string(gmtm->tm_mon+1);
    datetime += sep[1];
    datetime += std::to_string(gmtm->tm_mday);
    datetime += sep[2];
    datetime += std::to_string(gmtm->tm_hour);
    datetime += sep[3];
    datetime += std::to_string(gmtm->tm_min);
    datetime += sep[4];
    datetime += std::to_string(gmtm->tm_sec);

    return datetime;
}

/**
 * clamp value between min and max
 * @tparam T
 * @param val
 * @param min
 * @param max
 * @return
 */
template<typename T>
T clamp(T val, T min, T max) {
    if(val < min)
        return min;
    if(val > max)
        return max;
    return val;
}

/**
 * Load YAML node from file
 * @param conf_file
 * @return
 */
inline YAML::Node loadYAMLconf(std::string conf_file) {
    return YAML::LoadFile(conf_file);
}

/**
 * Get configuration from YAML node
 * @tparam T
 * @param conf yaml node
 * @param key configuration KEY
 * @param defaultVal defalt value in case of no KEY found
 * @return conf value
 */
template<typename T>
inline T getYAMLconf(YAML::Node conf, std::string key, T defaultVal) {
    T val = defaultVal;
    if(conf && conf[key]) {
        val = conf[key].as<T>();
    }
    //std::cout<<"YAML "<<key<<", val: "<<val<<"\n";
    return val;
}

/**
 *   Class for loop rate at a certain delta time
 *   in microsecs
 */
struct LoopRate {

    std::string name;
    timeStamp_t dt;
    timeStamp_t T, lastT;

    /**
     * init Looprate
     * @param _dt delta time in microsecs
     * @param _name name of the loop
     */
    LoopRate(timeStamp_t _dt, std::string _name = "anonymous") {
        dt  = _dt;
        name = _name;
        T = lastT = 0;
    }

    /**
     * Wait remaining time
     * it will print a warning on deadline miss
     */
    void wait(bool print = true) {
        T = getTimeStamp();

        timeStamp_t delta = T - lastT;
        if(delta >= 0 && delta < dt) {
            usleep(dt - delta);
        } else if(lastT > 0) {
            if(print)
                std::cout<<"LOOPRATE "<<name<<" exceeded: "<<delta<<" of "<<dt<<" ms\n";
        }

        lastT = getTimeStamp();
    }
};

/**
 * Cirular Array implementation
 * @tparam T
 */
template<typename T>
struct CircularArray {

    static const int MAX_DIM = 1000;
    T array[CircularArray::MAX_DIM];
    int dim, position;

    std::mutex m;

    /**
     * init circular array
     * @param _dim array dimension
     */
    CircularArray(int _dim = 100) {
        setDim(_dim);
        position = 0;
    }

    void clear() {
        position = 0;
    }

    /**
     * set array dimension
     * @param _dim
     */
    void setDim(int _dim) {
        if(dim > CircularArray::MAX_DIM)
            dim = CircularArray::MAX_DIM;
        dim = _dim;
    }

    /**
     * push an element to array
     * @param element
     */
    void add(T element) {
        std::lock_guard<std::mutex> lock(m);

        array[position % dim] = element;
        position++;
    }

    /**
     * get element from array head
     * @param n
     * @return
     */
    T head(int n) {
        std::lock_guard<std::mutex> lock(m);
        
        int index = position - 1 - n;
        index = (index + dim) % dim;
        return array[index];
    }

    /**
     * get maximum value without popping it
     * @return
     */
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

    /**
     * get minimum value without popping it
     * @return
     */
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

    /**
     * get number of elements in array
     * @return
     */
    int size() {
        if(position < dim)
            return position;
        else
            return dim;
    }

    /**
     * dump to file
     * @param file
     */
    void dump(std::string file) {     
        std::ofstream os;
        os.open(file);
        for(int i=0; i<size(); i++) {
            os<<head(i)<<"\n";
        }
        os.close();
    }
};

/**
 * Fifo queue implementation
 * @tparam T
 */
template<typename T>
struct FIFOQueue {

    static const int MAX_DIM = 1000;
    T array[FIFOQueue::MAX_DIM];
    int dim, inserted, position;

    std::mutex m;

    /**
     * init FIFO
     * @param _dim dimension
     */
    FIFOQueue(int _dim=1) {
        setDim(_dim);
        position = 0;
        inserted = 0;
    }

    /**
     * Set FIFO dimension
     * @param _dim
     */
    void setDim(int _dim) {
        if(dim > FIFOQueue::MAX_DIM)
            dim = FIFOQueue::MAX_DIM;
        dim = _dim;
    }

    /**
     * push element to FIFO
     * @param element
     */
    void add(T element) {
        std::lock_guard<std::mutex> lock(m);

        array[(position + inserted) % dim] = element;
        inserted++;
        if(inserted > dim)
            inserted = dim;
    }

    /**
     * pop element from FIFO
     * @param out output element
     * @return true if correctly popped
     */
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


/**
 * PoolQUEUE implementation
 * is is a pool of free elements ready to be used
 * is useful for use data structures and leave them
 * for later without deallocate
 * @tparam T
 */
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

    /**
     * Init Pool with dimension
     * @param _dim
     */
    PoolQueue(int _dim=1) {
        setDim(_dim);
        for(int i=0; i<MAX_DIM; i++) {
            mtx[i].unlock();
        }
    }

    /**
     * Set pool dimension
     * @param _dim
     */
    void setDim(int _dim) {
        if(_dim > PoolQueue::MAX_DIM)
            _dim = PoolQueue::MAX_DIM;
        dim = _dim;
    }

    /**
     * Add an element to pool
     * @param element
     * @return index for the added element, -1 on fail
     */
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

    /**
     * Get the index of the first free element
     * @param out
     * @return index, -1 on fail
     */
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

    /**
     * Release index
     * @param idx
     */
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
