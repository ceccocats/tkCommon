#pragma once

#include <iostream>
#include <string>
#include <execinfo.h>
#include <typeinfo>
#include <cxxabi.h>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>
#include "utils.h"

#define clsName(X) (tk::tformat::printErr(abi::__cxa_demangle(typeid(*this).name());

#define clsErr(X)  (tk::tformat::printErr(abi::__cxa_demangle(typeid(*this).name(), 0, 0, NULL),X));

#define clsSuc(X)  (tk::tformat::printSuc(abi::__cxa_demangle(typeid(*this).name(), 0, 0, NULL),X));

#define clsWrn(X)  (tk::tformat::printWrn(abi::__cxa_demangle(typeid(*this).name(), 0, 0, NULL),X));

#define clsMsg(X)  (tk::tformat::printMsg(abi::__cxa_demangle(typeid(*this).name(), 0, 0, NULL),X));

namespace tk { namespace tformat{

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
     * @brief                       Color formatting text
     * 
     * @param source                String to formatting
     * @param foreground_color      Foreground color
     * @param background_color      Background color
     * @return std::string          String Formatted
     */
    static std::string 
    print(std::string const& str, int foreground_color, int background_color = 39, int text_format = 39)
    {
        std::string const start = "\033[";
        std::string const end   = "\033[0m";

        background_color += 10;

        if(text_format == 30){
            text_format = 0;
        }

        return start + std::to_string(text_format) + std::string{";"} +
                std::to_string(foreground_color) + std::string{";"} + std::to_string(background_color) + std::string{"m"} + str + end;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief               Set class name in output format
     * 
     * @param s             Class name
     * @return std::string  Output stream
     */
    inline std::string
    formatName(std::string s){

        if(s.find("::") != -1){
            s = s.substr(s.find_last_of("::")+1);
        }

        s = std::string{"["} + s + std::string{"]:"};

        if(s.size() > LEN_CLASS_NAME)
            s.resize(LEN_CLASS_NAME);

        s += std::string(LEN_CLASS_NAME-s.size(), ' ');

        return s;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief               Set message in output format
     * 
     * @param s             Msg
     * @return std::string  Output stream
     */
    inline std::string
    formatMsg(std::string m){

        std::string out = "";

        struct winsize size;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);

        int len = size.ws_col - LEN_CLASS_NAME;
        if(len <= 0)
            len = 80;

        int c = 0;
        for(c = 0; c < (int)m.size(); c += len){

            out += m.substr(c,len) + '\n' + std::string(LEN_CLASS_NAME, ' ');
        }

        return out.substr(0,out.size()-1-LEN_CLASS_NAME);
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief           Error printing
     * 
     * @param className Class name
     * @param msg       message that need to print
     */
    inline void
    printErr(std::string className, std::string msg){

        msg = print(formatName(className),red) + formatMsg(msg);
        std::cerr<<msg;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief           Success printing
     * 
     * @param className Class name
     * @param msg       message that need to print
     */
    inline void
    printSuc(std::string className, std::string msg){

        msg = print(formatName(className),green) + formatMsg(msg);
        std::cout<<msg;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief           Warning printing
     * 
     * @param className Class name
     * @param msg       message that need to print
     */
    inline void
    printWrn(std::string className, std::string msg){

        msg = print(formatName(className),yellow) + formatMsg(msg);
        std::cout<<msg;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief           Message printing
     * 
     * @param className Class name
     * @param msg       message that need to print
     */
    inline void
    printMsg(std::string className, std::string msg){

        msg = print(formatName(className),predefined) + formatMsg(msg);
        std::cout<<msg;
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief                       Method that set terminal color
     * 
     * @param foreground_color      Foreground color
     * @param background_color      Background color
     * @return std::string          Set string
     */
    inline static std::string
    set(int foreground_color, int background_color = 39, int text_format = 39)
    {
        std::string const start = "\033[";

        background_color += 10;

        if(text_format == 30){
            text_format = 0;
        }

        return start + std::to_string(text_format) + std::string{";"} +
                std::to_string(foreground_color) + std::string{";"} + std::to_string(background_color) + std::string{"m"};
    }
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    /**
     * @brief                       Method that unset color
     * 
     * @return std::string          unset string 
     */
    static std::string
    unset()
    {
        return std::string("\033[0m");
    }
}}
