#pragma once 

#include <sstream>
#include <iostream>
#include <string>
#include <execinfo.h>
#include <typeinfo>
#include <cxxabi.h>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>

#define tkMSG(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::getInstance().log(tk::console::LogLevel::INFO,  __PRETTY_FUNCTION__, os);}
#define tkWRN(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::getInstance().log(tk::console::LogLevel::WARN,  __PRETTY_FUNCTION__, os);}
#define tkDBG(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::getInstance().log(tk::console::LogLevel::DEBUG, __PRETTY_FUNCTION__, os);}
#define tkERR(...) { std::stringstream os; os<<__VA_ARGS__; tk::console::Console::getInstance().log(tk::console::LogLevel::ERROR, __PRETTY_FUNCTION__, os);}

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
        Console() {
            verbosity = LogLevel::DEBUG;
        }

        std::string applyColor(std::string s, int foreground_color, int background_color = 39, int text_format = 39);
        std::string formatName(std::string s);
        std::string formatMsg(std::string s);

        LogLevel    verbosity;
    public:
        static Console& getInstance() 
        {
            static Console  instance;
            return instance;
        }

        Console(Console const&) = delete;
        void operator=(Console const&) = delete;

        void setVerbosity(LogLevel verbosityLevel);
        void log(const LogLevel level, const std::string &pretty, const std::stringstream &args);
    };
}}