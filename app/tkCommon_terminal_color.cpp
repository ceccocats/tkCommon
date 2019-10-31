#include <tkCommon/terminalColor.h>
#include <iostream>

int main(int argc, char* argv[]){

    std::cout<<tk::color::print("Questo è tk",tk::color::foreground::yellow)<<"\n\n";

    std::cout<<tk::color::print("Questo è tk",tk::color::foreground::yellow,tk::color::background::red)<<"\n\n";

    std::cout<<tk::color::print("Questo è tk","",tk::color::foreground::red)<<"\n\n";

    std::cout<<tk::color::print("Questo è tk","",tk::color::foreground::green)<<"\n\n";

    return  0;
}