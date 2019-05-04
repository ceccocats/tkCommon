#pragma once

#include <iostream>
#include <iomanip>
#include <ctime>
#include <map>
#include <vector>
#include <fstream>

// Colored output
#define COL_END "\033[0m"

#define COL_RED "\033[31m"
#define COL_GREEN "\033[32m"
#define COL_ORANGE "\033[33m"
#define COL_BLUE "\033[34m"
#define COL_PURPLE "\033[35m"
#define COL_CYAN "\033[36m"

#define COL_REDB "\033[1;31m"
#define COL_GREENB "\033[1;32m"
#define COL_ORANGEB "\033[1;33m"
#define COL_BLUEB "\033[1;34m"
#define COL_PURPLEB "\033[1;35m"
#define COL_CYANB "\033[1;36m"

// Simple Timer 
//#define TIMER_ENABLE
#ifdef TIMER_ENABLE 


    #define TIMER_START(name) timespec start##name, end##name;                         \
                              clock_gettime(CLOCK_MONOTONIC, &start##name);            

    #define TIMER_STOP_COL(name, col)  clock_gettime(CLOCK_MONOTONIC, &end##name);     \
        {double t_ns = ((double)(end##name.tv_sec - start##name.tv_sec) * 1.0e9 +      \
                    (double)(end##name.tv_nsec - start##name.tv_nsec))/1.0e6;          \
        std::cout<<col<<#name<<" Time:"<<std::setw(16)<<t_ns<<" ms\n"<<COL_END; }

    #define TIMER_STOP(name) TIMER_STOP_COL(name, COL_CYANB)

#else

    #define TIMER_START(name) ;
    #define TIMER_STOP(name) ;
    #define TIMER_DUMP ;
#endif
