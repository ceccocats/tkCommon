#include <tkCommon/terminalColor.h>
#include <iostream>
#include <cxxabi.h>

namespace tk{
class tkClass{
    public:
        void error(){
            clsErr("Messaggio di errore\n");
        }
        void success(){
            clsSuc("Messaggio di successo\n");
        }
        void warning(){
            clsWrn("Messaggio di warning\n");
        }
        void message(){
            clsMsg("Messaggio classico\n");
        }
};
}


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
    std::cout<<"Questo è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::yellow,tk::tcolor::red);
    std::cout<<"Questo è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red);
    std::cout<<"Questo è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red,tk::tcolor::bold);
    std::cout<<"Questo è tk"<<tk::tcolor::unset()<<"\n\n";

    std::cout<<tk::tcolor::set(tk::tcolor::predefined,tk::tcolor::red,tk::tcolor::reverse);
    std::cout<<"Questo è tk"<<tk::tcolor::unset()<<"\n\n";


    //Class printing
    tk::tkClass a;
    a.error();
    a.message();
    a.success();
    a.warning();

    return  0;
}