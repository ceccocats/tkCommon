#include <tkCommon/exceptions.h>
#include <iostream>

int sum_array(int *vals) {

    int sum = 0;

    for(int c = 0; c< 20000; c++){
        sum += vals[c];
    }
    return sum;
}

int main(int argc, char* argv[]){

    tk::exceptions::handleSegfault();

    int vals[10] = {0,1,2,3,4,5,6,7,8,9};

    int sum = sum_array(vals);
    std::cout<<"Somma: "<<sum<<std::endl;
    return 0;
}