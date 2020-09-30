#include <tkCommon/printTerminal.h>
#include <iostream>
#include <cxxabi.h>

namespace tk{
class tkClass{
    public:
        void error(){
            clsErr("error message\n");
        }
        void success(){
            clsSuc("success message\n");
        }
        void warning(){
            clsWrn("warning message\n");
        }
        void message(){
            clsMsg("classic message. Each type message print reading the terminal column in a way that a single print remain at left of class name. This permits to have a clean print. I don't konw what'else to write for have a long phrase for demostrating the work. Bye Bye\n");
        }
};
}


int main(int argc, char* argv[]){

    std::cout<<"this is how "
             <<tk::tprint::print("colorful", tk::tprint::lightRed, tk::tprint::predefined, tk::tprint::underlined)
             <<" is "
             <<tk::tprint::print("tk:\n\n", tk::tprint::lightYellow, tk::tprint::predefined, tk::tprint::bold);

    //Print usage
    for(uint8_t i=tk::tprint::white; i<=tk::tprint::lightGray; i++) {
        std::cout<<tk::tprint::print("Color code:  ", i);
        std::cout<<tk::tprint::print(std::to_string(i)+"\n", i, tk::tprint::predefined, tk::tprint::reverse);
    }

    //Set unset usage
    for(uint8_t i=tk::tprint::darkGray; i<=tk::tprint::black; i++) {
        std::cout<<tk::tprint::set(i);
        std::cout<<"Color code:  ";
        std::cout<<tk::tprint::set(i, tk::tprint::predefined, tk::tprint::reverse);
        std::cout<<int(i)<<tk::tprint::unset()<<"\n";
    }

    //Class printing
    tk::tkClass a;
    a.error();
    a.message();
    a.success();
    a.warning();

    return  0;
}