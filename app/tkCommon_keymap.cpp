#include <tkCommon/common.h>
#include <tkCommon/rt/Profiler.h>

int main(int argc, char* argv[]){

    for(int limit = 1e4; limit < 1e7; limit*=10) {

        // c++ implementation with string Key 
        std::map<std::string, float> mapcxx;
        mapcxx["ciao1"] = 0;
        mapcxx["ciao2"] = 0;
        mapcxx["ciao3"] = 0;
        mapcxx["ciao4"] = 0;
        
        tkPROF_tic(cxxMap)
        for(int i=0; i<limit; i++) {
            mapcxx["ciao1"] = i*1;
            mapcxx["ciao2"] = i*2;
            mapcxx["ciao3"] = i*3;
            mapcxx["ciao4"] = i*4;
        }
        tkPROF_toc(cxxMap)
        
        // c++ implementation with int Key 
        std::map<int, float> mapcxxInt;
        mapcxxInt[0] = 0;
        mapcxxInt[1] = 0;
        mapcxxInt[2] = 0;
        mapcxxInt[3] = 0;
        tkPROF_tic(mapcxxInt)
        for(int i=0; i<limit; i++) {
            mapcxxInt[0] = i*1;
            mapcxxInt[1] = i*2;
            mapcxxInt[2] = i*3;
            mapcxxInt[3] = i*4;
        }
        tkPROF_toc(mapcxxInt)

        // ASSERTS inside [] check make it slow 
        // tk implementation with string converted to int at compile time
        tk::common::Map<float> map;
        // tell the key -> string association
        // is not mandatory, but if we dont insert the key the Map cant retrive the original string
        map.add("ciao1");
        map.add("ciao2");
        map.add("ciao3");
        map.add("ciao4");
        tkPROF_tic(tkMap)
        for(int i=0; i<limit; i++) {
            map[tkKey("ciao1")] = i*1;
            map[tkKey("ciao2")] = i*2;
            map[tkKey("ciao3")] = i*3;
            map[tkKey("ciao4")] = i*4;
        }
        tkPROF_toc(tkMap)

        //map.print();   
        std::cout<<"\n";
    }
    return  0;
}