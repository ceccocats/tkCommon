#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <mutex>
#include <sys/time.h>
#include <unistd.h>

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
    struct timespec tp;
    clock_gettime(CLOCK_REALTIME, &tp);
    return timeStamp_t(tp.tv_sec)*1e6 + tp.tv_nsec/1000;
}

/**
 * Convert timeval to microseconds from epoch
 * @param tv
 * @return
 */
inline timeStamp_t tv2TimeStamp(struct timeval tv) {
    return timeStamp_t(tv.tv_sec)*1e6 + tv.tv_usec;
} 

/**
 * Convert timespac to microseconds from epoch
 * @param tv
 * @return
 */
inline timeStamp_t tp2TimeStamp(struct timespec tp) {
    return timeStamp_t(tp.tv_sec)*1e6 + tp.tv_nsec/1000;
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