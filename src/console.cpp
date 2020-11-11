#include "tkCommon/console.h"

namespace tk { namespace console {
    void 
    Console::log(const LogLevel level, const std::string &pretty, const std::stringstream &args) { 
        if (level < verbosity)
            return;

        int color;
        switch (level)
        {
        case LogLevel::INFO:
            color = black;
            break;
        case LogLevel::DEBUG:
            color = lightGray;
            break;
        case LogLevel::WARN:
            color = yellow;
            break;
        case LogLevel::ERROR:
            color = red;
            break;
        default:
            color = white;
            break;
        }


        std::cout<<applyColor(formatName(pretty), color)<<formatMsg(args.str());
    };

    void 
    Console::setVerbosity(LogLevel verbosityLevel) {
        verbosity = verbosityLevel;
    }

    std::string
    Console::applyColor(std::string s, int foreground_color, int background_color, int text_format) {
        std::string const start = "\033[";
        std::string const end   = "\033[0m";

        background_color += 10;

        if(text_format == 30){
            text_format = 0;
        }

        return start + std::to_string(text_format) + std::string{";"} +
                std::to_string(foreground_color) + std::string{";"} + std::to_string(background_color) + std::string{"m"} + s + end;
    }

    std::string 
    Console::formatName(std::string s) {
        // remove parameters and return type
        int start   = s.find_first_of('(');
        int len     = start - s.find_last_of(')');
        s.erase(start, len);
        s.erase(0, s.find_last_of(' '));

        if(s.size() > LEN_CLASS_NAME)
            s.resize(LEN_CLASS_NAME);

        s += std::string(LEN_CLASS_NAME-s.size(), ' ');

        return s;
    }
    
    std::string 
    Console::formatMsg(std::string s) {
        std::string out = "";

        struct winsize size;
        ioctl(STDOUT_FILENO, TIOCGWINSZ, &size);

        int len = size.ws_col - LEN_CLASS_NAME;
        if(len <= 0)
            len = 80;

        int c = 0;
        for(c = 0; c < (int)s.size(); c += len){

            out += s.substr(c,len) + '\n' + std::string(LEN_CLASS_NAME, ' ');
        }

        return out.substr(0,out.size()-1-LEN_CLASS_NAME);
    }
}}