#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
#include <regex>
#include <tkCommon/terminalFormat.h>

#define tkASSERT(X) tk::exceptions::check_error(X,__FILE__,__FUNCTION__,__LINE__);

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

                    tk::tformat::printErr("Exception",std::string{"Error at line "}+std::to_string(line)+" in funtion "+funz+" at "+file+"\n");
                    exit(-1);
                }
            }
            
        private:

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


                std::cout<<tk::tformat::set(tk::tformat::red,tk::tformat::predefined,tk::tformat::bold);
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
                    int e = system(cmd.c_str());
                }
                std::cerr<<"----------------------------------------------------------------------";
                std::cout<<tk::tformat::unset()<<"\n";

                free (strings);

                exit(-1);
            }
    };
}