#include <tkCommon/terminalColor.h>
#include <iostream>

int main(int argc, char* argv[]){

    //Print usage
    std::cout<<"Using print:\n";

    std::cout<<tk::tcolor::print("Questo è tk",tk::tcolor::yellow)<<"\n\n";

    std::cout<<tk::tcolor::print("Questo è tk",tk::tcolor::yellow,tk::tcolor::red)<<"\n\n";

    std::cout<<tk::tcolor::print("Questo è tk",tk::tcolor::predefined,tk::tcolor::red)<<"\n\n";

    std::cout<<tk::tcolor::print("Questo è tk",tk::tcolor::predefined,tk::tcolor::predefined,tk::tcolor::bold)<<"\n\n";

    std::cout<<tk::tcolor::print("Questo è tk",tk::tcolor::predefined,tk::tcolor::predefined,tk::tcolor::reverse)<<"\n\n";


    //Set unset usage
    std::cout<<"Using set/unset:\n";

    std::cout<<tk::tcolor::set(tk::tcolor::yellow);
    std::cout<<"Qusto è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::yellow,tk::tcolor::red);
    std::cout<<"Qusto è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red);
    std::cout<<"Qusto è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red,tk::tcolor::bold);
    std::cout<<"Qusto è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red,tk::tcolor::reverse);
    std::cout<<"Qusto è tk"<<tk::tcolor::unset()<<"\n\n";



    return  0;
}