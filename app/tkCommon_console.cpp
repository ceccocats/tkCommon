#include "tkCommon/console.h"

int main( int argc, char** argv){
    tk::console::Console::get().enableFileLog();

    int ciao = 1000;
    tkMSG("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkWRN("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkERR("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    tkDBG("ciao "<<ciao<<" sorrata "<<ciao<<"\n");
    
    return 0;
}