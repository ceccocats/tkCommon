#pragma once

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <regex>
#include <fstream>
#include <tkCommon/terminalFormat.h>

#define tkASSERT(X) tk::exceptions::check_error(X,__FILE__,__FUNCTION__,__LINE__);
#define tkFATAL(X) tk::exceptions::raise_error(X,__FILE__,__FUNCTION__,__LINE__);

namespace tk{

    class exceptions{

        public:

            static void handleSegfault(){

                struct sigaction sa;
                memset(&sa, 0, sizeof(struct sigaction));
                sigemptyset(&sa.sa_mask);
                sa.sa_sigaction = sign_handler;
                sa.sa_flags   = SA_SIGINFO;

                sigaction(SIGSEGV, &sa, NULL);
            }

            inline static void check_error(bool status,const char *file, const char *funz, int line){
                
                if(status == false){

                    tk::tformat::printErr("tkAssert",std::string{"Error at line "}+std::to_string(line)+" in funtion "+funz+" at "+file+"\n");
                    sign_handler(0, 0, 0);
                    exit(-1);
                }
            }

            inline static void raise_error(std::string msg, const char *file, const char *funz, int line) {
                std::cout<<msg<<"\n";
                std::cout<<"file: "<<file<<" function: "<<funz<<" line: "<<line<<"\n";
                exit(-1);
            }
            
        private:

            static std::string ssystem (std::string command) {
                std::string tmpname = "temp.txt";
                std::string scommand = command;
                std::string cmd = scommand + " >> " + tmpname;
                std::system(cmd.c_str());
                std::ifstream file(tmpname, std::ios::in | std::ios::binary );
                std::string result;
                if (file) {
                    while (!file.eof()) result.push_back(file.get());
                    file.close();
                }
                remove(tmpname.c_str());
                return result.substr(0,(int)result.size()-2);
            }

            static void sign_handler(int sig, siginfo_t *dont_care, void *dont_care_either){

                void *array[16];
                size_t size;
                char **strings;
                size_t i;

                size = backtrace (array, 16);
                strings = backtrace_symbols (array, size);

                // get path of exec
                char name_buf[2048];
                name_buf[::readlink("/proc/self/exe", &name_buf[0], 2047)] = 0;

                // regex address at the end of trace string
                std::regex addr_reg("\\[(0x.*?)\\]"); // inside brackets []


                std::cout<<tk::tformat::set(tk::tformat::red,tk::tformat::predefined,tk::tformat::bold)<<"\n";
                std::cerr<<"SEGFAULT\n";
                std::cerr<<"----------------------------------------------------------------------\n";
                for (i = 0; i < size; i++){
                    std::cerr<<strings[i]<<"\n";

                    // get address and call add2line to get the line in code
                    std::string addr(strings[i]);
                    std::smatch m;
                    std::regex_search(addr, m, addr_reg);
                    for (auto x : m)
                        addr = x;
                    std::string cmd = std::string("addr2line -e ") + name_buf + " " + addr;
                    std::string output = ssystem(cmd);

                    if(output.find("?") != 0 && i != 0){

                        std::string file = output.substr(0,output.find(":"));
                        std::string line = output.substr(output.find(":")+1);
                        line = line.substr(0,line.find("(")-1);

                        std::cout<<tk::tformat::set(tk::tformat::predefined,tk::tformat::predefined,tk::tformat::bold)<<"\n";
                        output = "sed -n "+line+"p "+file;
                        std::cout<<"Error in file "<<file<<" at line "<<line<<"\n\n";
                        int p = system(output.c_str());
                        std::cout<<tk::tformat::set(tk::tformat::red,tk::tformat::predefined,tk::tformat::bold)<<"\n";

                    }                    
                }
                std::cerr<<"----------------------------------------------------------------------";
                std::cout<<tk::tformat::unset()<<"\n";

                free (strings);

                exit(-1);
            }
    };
}
