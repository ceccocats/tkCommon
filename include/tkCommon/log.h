#pragma once 

#include <sstream>
#include <iostream>
#include <fstream>
#include <string>
#include <execinfo.h>
#include <typeinfo>
#include <cxxabi.h>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>

#define tkMSG(...) { std::stringstream os; os<<__VA_ARGS__; tk::Log::get().log(tk::LogLevel::INFO,  __PRETTY_FUNCTION__, os.str());}
#define tkWRN(...) { std::stringstream os; os<<__VA_ARGS__; tk::Log::get().log(tk::LogLevel::WARN,  __PRETTY_FUNCTION__, os.str());}
#define tkDBG(...) { std::stringstream os; os<<__VA_ARGS__; tk::Log::get().log(tk::LogLevel::DEBUG, __PRETTY_FUNCTION__, os.str());}
#define tkERR(...) { std::stringstream os; os<<__VA_ARGS__; tk::Log::get().log(tk::LogLevel::ERROR, __PRETTY_FUNCTION__, os.str());}

namespace tk {
    /**
     * 
     */
    enum class LogLevel {
        DEBUG,
        INFO,
        WARN,
        ERROR,
        FATAL 
    };

    enum class LogTextFormat : uint8_t {
        predefined      = 39,
        white           = 30,
        red             = 31,
        green           = 32,
        yellow          = 33,
        blue            = 34,
        magenta         = 35,
        cyan            = 36,
        lightGray       = 37,
        darkGray        = 90,
        lightRed        = 91,
        lightGreen      = 92,
        lightYellow     = 93,
        lightBlue       = 94,
        lightMagenta    = 95,
        lightCyan       = 96,
        black           = 97,
        bold            = 1,
        dim             = 2,
        underlined      = 4,
        blink           = 5,
        reverse         = 7,
        hidden          = 8
    };  
    
    /**
     * 
     */
    class Log {
    private:
        /**
         * 
         */
        Log() {
            verbosity       = LogLevel::DEBUG;
            startStrFormat  = "\033[";
            endStrFormat    = "\033[0m";
            fileLogEnabled  = false;
        }

        /**
         * 
         */
        std::string applyColor(const std::string &s, const LogLevel level);
        
        /**
         * 
         */
        std::string formatName(const std::string &s);
        
        /**
         * 
         */
        std::string formatMsg(const std::string &s);

        LogLevel        verbosity;
        
        bool            fileLogEnabled;
        std::ofstream   fileLog;

        bool            filterEnabled;
        std::string     filterStr;

        std::string     startStrFormat;
        std::string     endStrFormat;
        struct winsize  winSize;

        static const int CLASS_NAME_LEN = 32;
    public:
        /**
         * 
         */
        static Log& get() 
        {
            static Log  instance;
            return instance;
        }

        ~Log();

        Log(Log const&) = delete;
        void operator=(Log const&) = delete;

        /**
         * 
         */
        void setVerbosity(const LogLevel verbosityLevel);

        /**
         * 
         */
        void setFileLog(const std::string &fileName);

        /**
         * 
         */
        void disableFileLog();

        /**
         * 
         */
        void setFilter(const std::string &filter);

        /**
         * 
         */
        void log(const LogLevel level, const std::string &pretty, const std::string &args);
    };
}