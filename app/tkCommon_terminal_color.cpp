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

    std::cout<<"this is how "
             <<tk::tcolor::print("colorful", tk::tcolor::lightRed, tk::tcolor::predefined, tk::tcolor::underlined)
             <<" is "
             <<tk::tcolor::print("tk:\n\n", tk::tcolor::lightYellow, tk::tcolor::predefined, tk::tcolor::bold);

    //Print usage
    for(uint8_t i=tk::tcolor::white; i<=tk::tcolor::lightGray; i++) {
        std::cout<<tk::tcolor::print("Color code:  ", i);
        std::cout<<tk::tcolor::print(std::to_string(i)+"\n", i, tk::tcolor::predefined, tk::tcolor::reverse);
    }

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