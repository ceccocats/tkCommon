#include "tkCommon/log.h"

namespace tk { namespace nonloso {
    class Test {
        public:
            Test() {
                tkERR("non sono capace\n");
            }
            ~Test() = default;
    };
}}


int main( int argc, char** argv){
    tk::Log::get().setFileLog("log.tast");
    //tk::Log::get().setFilter("e");

    tk::nonloso::Test bh;

    int ciao = 1000;
    tkMSG("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkWRN("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkERR("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkDBG("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    
    return 0;
}