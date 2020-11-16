#pragma once

#include <iostream>
#include <string>
#include <execinfo.h>
#include <typeinfo>
#include <iomanip>
#include <sys/ioctl.h>
#include <unistd.h>

namespace tk { namespace term {

    /**
     *  Terminal ProgressBar
     * 
     *  Example usage:
     *  tk::tprint::ProgressBar pbar(0, "example progress");
     *  for(;pbar.eval(1000); pbar.i()++) {
     *      usleep(5000);
     *  }
     */ 
    class ProgressBar {
    private:
        int _end = 0;
        int _idx = 0;
        std::string _name;

    public:
        ProgressBar(int start, std::string name = "") {
            _idx = start;
            _name = name;
        }
        int size() {
            return _end;
        }
        int &i() {
            return _idx;
        }
        bool eval(int end) {
            _end = end;
            bool alive = _idx < end;

            struct winsize ws;
            ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws);
            
            std::string left_str = _name + " |";
            std::string right_str = "| " + std::to_string(_idx) + "/" + std::to_string(_end);

            int dim = ws.ws_col - left_str.size() - right_str.size();
            std::cout<<left_str;
            int endbar = (float(_idx) / _end)*dim;
            for(int i=0; i<dim; i++) {
                i<=endbar ? std::cout<<"#" : std::cout<<" ";
            }
            std::cout<<right_str;
            
            if(alive) 
                std::cout<<"\r"<<std::flush;
            else
                std::cout<<"\n";
            return alive;
        }
    };



    /**
     * dump memory as hexdump
     * @param os
     * @param buffer
     * @param bufsize
     * @param showPrintableChars
     * @return
     */

    inline std::ostream& hex_dump(std::ostream& os, const void *buffer,
                                  std::size_t bufsize, bool showPrintableChars = true) {
        if (buffer == nullptr) {
            return os;
        }
        auto oldFormat = os.flags();
        auto oldFillChar = os.fill();
        constexpr std::size_t maxline{8};
        // create a place to store text version of string
        char renderString[maxline+1];
        char *rsptr{renderString};
        // convenience cast
        const unsigned char *buf{reinterpret_cast<const unsigned char *>(buffer)};

        for (std::size_t linecount=maxline; bufsize; --bufsize, ++buf) {
            os << std::setw(2) << std::setfill('0') << std::hex
               << static_cast<unsigned>(*buf) << ' ';
            *rsptr++ = std::isprint(*buf) ? *buf : '.';
            if (--linecount == 0) {
                *rsptr++ = '\0';  // terminate string
                if (showPrintableChars) {
                    os << " | " << renderString;
                }
                os << '\n';
                rsptr = renderString;
                linecount = std::min(maxline, bufsize);
            }
        }
        // emit newline if we haven't already
        if (rsptr != renderString) {
            if (showPrintableChars) {
                for (*rsptr++ = '\0'; rsptr != &renderString[maxline+1]; ++rsptr) {
                    os << "   ";
                }
                os << " | " << renderString;
            }
            os << '\n';
        }

        os.fill(oldFillChar);
        os.flags(oldFormat);
        return os;
    }

}}
