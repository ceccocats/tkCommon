#pragma once

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <regex>
#include <fstream>
#include <tkCommon/printTerminal.h>


#define tkASSERT(...) tk::exceptions::check_error(__FILE__,__FUNCTION__,__LINE__,__VA_ARGS__);
#define tkFATAL(X) tk::exceptions::raise_error(__FILE__,__FUNCTION__,__LINE__,X);
#define tkUNIT_TEST(...) tk::exceptions::check_unit_test(__FILE__,__FUNCTION__,__LINE__,__VA_ARGS__);

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

            inline static void check_error(const char *file, const char *funz, int line, bool status, std::string msg = ""){
                
                if(status == false){
                    if(msg != "")
                        tk::tprint::printErr("tkAssert", msg+"\n");
                    tk::tprint::printErr("tkAssert", "function: "+std::string(funz)+" at "+file+":"+ std::to_string(line)+"\n");
                    exit(-1);
                }
            }

            inline static void raise_error(const char *file, const char *funz, int line, std::string msg) {
                tk::tprint::printErr("tkFatal", msg+"\nfunction: "+std::string(funz)+" at "+file+":"+ std::to_string(line)+"\n");
                exit(-1);
            }

            inline static void check_unit_test(const char *file, const char *funz, int line, bool status, std::string msg = ""){
                
                if(status == false){
                    if(msg != "")
                        tk::tprint::printErr("tkUNIT_TEST", msg+"\n");
                    tk::tprint::printErr("tkUNIT_TEST", "failed at: "+std::string(funz)+" at "+file+":"+ std::to_string(line)+"\n");
                }
            }
        
        private:

            static std::string ssystem (std::string command) {
                std::string tmpname = "temp.txt";
                std::string scommand = command;
                std::string cmd = scommand + " >> " + tmpname;
                int err = std::system(cmd.c_str());
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


                std::cout<<tk::tprint::set(tk::tprint::red,tk::tprint::predefined,tk::tprint::bold)<<"\n";
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

                        std::cout<<tk::tprint::set(tk::tprint::predefined,tk::tprint::predefined,tk::tprint::bold)<<"\n";
                        output = "sed -n "+line+"p "+file;
                        std::cout<<"Error in file "<<file<<" at line "<<line<<"\n\n";
                        int p = system(output.c_str());
                        std::cout<<tk::tprint::set(tk::tprint::red,tk::tprint::predefined,tk::tprint::bold)<<"\n";

                    }                    
                }
                std::cerr<<"----------------------------------------------------------------------";
                std::cout<<tk::tprint::unset()<<"\n";

                free (strings);

                exit(-1);
            }
    };
}
