#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <execinfo.h>
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

                void *array[10];
                size_t size;
                char **strings;
                size_t i;

                size = backtrace (array, 10);
                strings = backtrace_symbols (array, size);

                std::cout<<tk::tformat::set(tk::tformat::predefined,tk::tformat::red);
                std::cerr<<"SEGFAULT\n";
                std::cerr<<"----------------------------------------------------------------------\n";
                for (i = 0; i < size; i++){
                    std::cerr<<strings[i]<<"\n";
                }
                std::cerr<<"----------------------------------------------------------------------";  
                std::cout<<tk::tformat::unset()<<"\n";        

                free (strings);

                exit(-1);
            }
    };
}