#pragma once

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <regex>
#include <fstream>
#include "tkCommon/log.h"

#define tkASSERT(...)    tk::exceptions::check_error(__FILE__,__FUNCTION__,__LINE__,__VA_ARGS__);
#define tkFATAL(X)       tk::exceptions::raise_error(__FILE__,__FUNCTION__,__LINE__,X);

namespace tk{ namespace exceptions{


inline static void check_error(const char *file, const char *funz, int line, bool status, std::string msg = ""){
    
    if(status == false){
        if(msg != "")
            tkERR("tkAssert", msg+"\n");
        tkERR("tkAssert", "function: "+std::string(funz)+" at "+file+":"+ std::to_string(line)+"\n");
        throw std::runtime_error("tkAssert");
    }
}

inline static void raise_error(const char *file, const char *funz, int line, std::string msg) {
    tkERR("tkFatal", msg+"\nfunction: "+std::string(funz)+" at "+file+":"+ std::to_string(line)+"\n");
    throw std::runtime_error("tkFatal");
}

}}
