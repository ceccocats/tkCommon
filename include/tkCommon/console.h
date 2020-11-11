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

#define tkMSG(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::get().log(tk::console::LogLevel::INFO,  __PRETTY_FUNCTION__, os.str());}
#define tkWRN(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::get().log(tk::console::LogLevel::WARN,  __PRETTY_FUNCTION__, os.str());}
#define tkDBG(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::get().log(tk::console::LogLevel::DEBUG, __PRETTY_FUNCTION__, os.str());}
#define tkERR(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::get().log(tk::console::LogLevel::ERROR, __PRETTY_FUNCTION__, os.str());}

namespace tk { namespace console {
    /**
     * 
     */
    enum class LogLevel{
        DEBUG,
        INFO,
        WARN,
        ERROR,
        FATAL 
    };

    const static uint8_t    predefined      = 39;
    const static uint8_t    white           = 30;
    const static uint8_t    red             = 31;
    const static uint8_t    green           = 32;
    const static uint8_t    yellow          = 33;
    const static uint8_t    blue            = 34;
    const static uint8_t    magenta         = 35;
    const static uint8_t    cyan            = 36;
    const static uint8_t    lightGray       = 37;
    const static uint8_t    darkGray        = 90;
    const static uint8_t    lightRed        = 91;
    const static uint8_t    lightGreen      = 92;
    const static uint8_t    lightYellow     = 93;
    const static uint8_t    lightBlue       = 94;
    const static uint8_t    lightMagenta    = 95;
    const static uint8_t    lightCyan       = 96;
    const static uint8_t    black           = 97;

    const static uint8_t    bold            = 1;
    const static uint8_t    dim             = 2;
    const static uint8_t    underlined      = 4;
    const static uint8_t    blink           = 5;
    const static uint8_t    reverse         = 7;
    const static uint8_t    hidden          = 8;

    const static int        LEN_CLASS_NAME  = 16;
    
    /**
     * 
     */
    class Console
    {
    private:
        /**
         * 
         */
        Console() {
            verbosity       = LogLevel::DEBUG;
            startStrFormat  = "\033[";
            endStrFormat    = "\033[0m";
            fileLogEnabled  = false;
        }

        /**
         * 
         */
        std::string applyColor(const std::string &s, uint8_t textColor, uint8_t bgColor = predefined, uint8_t textFormat = predefined);
        
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

        std::string     startStrFormat;
        std::string     endStrFormat;
        struct winsize  winSize;
    public:
        /**
         * 
         */
        static Console& get() 
        {
            static Console  instance;
            return instance;
        }

        ~Console();

        Console(Console const&) = delete;
        void operator=(Console const&) = delete;

        /**
         * 
         */
        void setVerbosity(LogLevel verbosityLevel);

        /**
         * 
         */
        void enableFileLog(std::string fileName = "");

        /**
         * 
         */
        void disableFileLog();

        /**
         * 
         */
        void log(const LogLevel level, const std::string &pretty, const std::string &args);
    };
}}