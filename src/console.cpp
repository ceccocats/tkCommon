#include "tkCommon/console.h"

namespace tk { namespace console {
    Console::~Console() {
        if (fileLogEnabled)
            fileLog.close();
    }

    void 
    Console::log(const LogLevel level, const std::string &pretty, const std::string &args) {        
        if (level < verbosity)
            return;

        std::string msg     = formatMsg(args);
        std::string name    = formatName(pretty); 

        if (fileLogEnabled)
            fileLog<<name<<msg;

        // color name for termina output
        uint8_t color;
        switch (level) {
        case LogLevel::INFO:
            color = black;
            break;
        case LogLevel::DEBUG:
            color = cyan;
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
        name = applyColor(name, color);

        
        
        std::cout<<name<<msg;
    };

    void 
    Console::setVerbosity(LogLevel verbosityLevel) {
        verbosity = verbosityLevel;
    }

    void 
    Console::enableFileLog(std::string fileName) {
        if (!fileLogEnabled) {
            if (fileName.compare("") == 0)
            fileLog.open("log.txt");
            else 
                fileLog.open(fileName);
            
            fileLogEnabled = true;
        }
    }

    void Console::disableFileLog() {
        if (fileLogEnabled) {
            fileLog.close();
            fileLogEnabled = false;
        }
    }

    std::string
    Console::applyColor(const std::string &s, uint8_t textColor, uint8_t bgColor, uint8_t textFormat) {
        bgColor+=10;
        if(textFormat == 30)
            textFormat = 0;

        return startStrFormat + std::to_string(textFormat) + std::string{";"} +
               std::to_string(textColor) + std::string{";"} + std::to_string(bgColor) + std::string{"m"} + s + endStrFormat;
    }

    std::string 
    Console::formatName(const std::string &s) {
        std::string out = "";
        
        int start   = s.find_last_of(' ', s.find_first_of('('));
        int len     = s.find_first_of('(') - start;
        out         += s.substr(start, len);

        if(out.size() > LEN_CLASS_NAME)
            out.resize(LEN_CLASS_NAME);

        out += std::string(LEN_CLASS_NAME - out.size(), ' ');

        return out;
    }
    
    std::string 
    Console::formatMsg(const std::string &s) {
        std::string out = "";

        ioctl(STDOUT_FILENO, TIOCGWINSZ, &winSize);

        int len = winSize.ws_col - LEN_CLASS_NAME;
        if(len <= 0)
            len = 80;

        int c = 0;
        for(c = 0; c < (int)s.size(); c += len){

            out += s.substr(c,len) + '\n' + std::string(LEN_CLASS_NAME, ' ');
        }

        return out.substr(0,out.size()-1-LEN_CLASS_NAME);
    }
}}