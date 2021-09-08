#include "tkCommon/log.h"

namespace tk {
    Log::~Log() {
        if (fileLogEnabled)
            fileLog.close();
    }

    void 
    Log::log(const LogLevel level, const std::string &pretty, const std::string &args) {        
        if (level < verbosity)
            return;

        std::string msg     = formatMsg(args);
        std::string name    = formatName(pretty); 

        // newline if not already inserted
        if(msg.back() != '\n')
            msg.push_back('\n');

        // log to file if enabled
        if (fileLogEnabled)
            fileLog<<name<<msg;

        // apply filter if enabled
        if (filterEnabled) {
            if (name.find(filterStr) == std::string::npos)
                return;
        }

        // color the name for the terminal
        name = applyColor(name, level);
        
        #if TKROS_VERSION == 2
            std::cerr<<name<<msg; // on ros2 cout is buffered, cerr not.
                                  // using cerr we can see tk logs even on ros2
        #else
            std::cout<<name<<msg;    
        #endif
    };

    void 
    Log::setVerbosity(const LogLevel verbosityLevel) {
        verbosity = verbosityLevel;
    }

    void 
    Log::setFileLog(const std::string &fileName) {
        if (!fileLogEnabled) {
            bool ok = false; 
            if (fileName.substr(fileName.find_last_of(".") + 1) == "txt") {
                fileLog.open(fileName);
                fileLogEnabled = true;
            } else {
                log(LogLevel::ERROR, "tk::Log::log", "Wrong file extension, use .txt\n");
            }
        }
    }

    void Log::disableFileLog() {
        if (fileLogEnabled) {
            fileLog.close();
            fileLogEnabled = false;
        }
    }

    void 
    Log::setFilter(const std::string &filter) {
        filterStr     = filter;
        filterEnabled = true;
    }

    std::string
    Log::applyColor(const std::string &s, const LogLevel level) {
        uint8_t textColor; 
        uint8_t bgColor     = (uint8_t) LogTextFormat::predefined;
        uint8_t textFormat  = (uint8_t) LogTextFormat::predefined;
        
        switch (level) {
            case LogLevel::INFO:
                textColor = (uint8_t) LogTextFormat::black;
                break;
            case LogLevel::DEBUG:
                textColor = (uint8_t) LogTextFormat::cyan;
                break;
            case LogLevel::WARN:
                textColor = (uint8_t) LogTextFormat::yellow;
                break;
            case LogLevel::ERROR:
                textColor = (uint8_t) LogTextFormat::red;
                break;
            default:
                textColor = (uint8_t) LogTextFormat::black;
                break;
        }

        bgColor += 10;
        if(textFormat == 30)
            textFormat = 0;

        return startStrFormat + std::to_string(textFormat) + std::string{";"} +
               std::to_string(textColor) + std::string{";"} + std::to_string(bgColor) + std::string{"m"} + s + endStrFormat;
    }

    std::string 
    Log::formatName(const std::string &s) {
        std::string out     = "";
        
        std::size_t start   = s.find_last_of(' ', s.find_first_of('('));
        if (start == std::string::npos)
            start = 0;
        else 
            start += 1;
        std::size_t len     = s.find_first_of('(') - start;
        
        out += s.substr(start, len);
        if(out.size() > CLASS_NAME_LEN)
            out.resize(CLASS_NAME_LEN);

        out += std::string(CLASS_NAME_LEN - out.size(), ' ');

        return out;
    }
    
    std::string 
    Log::formatMsg(const std::string &s) {
        std::string out = "";

        ioctl(STDOUT_FILENO, TIOCGWINSZ, &winSize);

        int len = winSize.ws_col - CLASS_NAME_LEN;
        if(len <= 0)
            len = 80;

        int c = 0;
        for(c = 0; c < (int)s.size(); c += len){

            out += s.substr(c,len) + '\n' + std::string(CLASS_NAME_LEN, ' ');
        }

        return out.substr(0,out.size()-1-CLASS_NAME_LEN);
    }
}