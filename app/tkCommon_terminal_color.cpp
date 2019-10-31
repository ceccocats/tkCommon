#include <tkCommon/terminalColor.h>
#include <iostream>

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
    for(uint8_t i=tk::tcolor::darkGray; i<=tk::tcolor::black; i++) {
        std::cout<<tk::tcolor::set(i);
        std::cout<<"Color code:  ";
        std::cout<<tk::tcolor::set(i, tk::tcolor::predefined, tk::tcolor::reverse);
        std::cout<<int(i)<<tk::tcolor::unset()<<"\n";
    }
    return  0;
}