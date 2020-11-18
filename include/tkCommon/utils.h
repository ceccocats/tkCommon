#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <sys/time.h>
#include <unistd.h>
#include "yaml-cpp/yaml.h" 

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
 * Cirular Array implementation
 * @tparam T
 */
template<typename T>
struct CircularArray {

    T *array = nullptr;
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

    ~CircularArray() {
        if(array != nullptr)
            delete [] array;
    }

    void clear() {
        position = 0;
    }

    /**
     * set array dimension
     * @param _dim
     */
    void setDim(int _dim) {
        if(array != nullptr){
            T* temp = new T[_dim];
            memcpy(temp,array,std::min(dim,_dim)*sizeof(T));
            delete[] array;
            array = temp;
        }else{
            array = new T[_dim];
        }
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
 * @brief           Function for convert degrees in radiants
 * 
 * @param x         degrees
 * @return double   degrees in radiant
 */
inline double toRadians(double x){
    
    return ((x * M_PI) / 180.0);
}

inline std::vector<std::string> splitString(const std::string &s, char delim, int maxelems = -1) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string substr;
    while(std::getline(ss, substr, delim) ) {
        elems.push_back(substr);
        if(maxelems > 0 && elems.size() >= maxelems)
            break;
    }
    return elems;
}
