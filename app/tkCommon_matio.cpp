#define TIMER_ENABLE
#include "tkCommon/CmdParser.h"
#include "tkCommon/math/MatIO.h"
#include <thread>
#include <signal.h>

class TestDump : public tk::math::MatDump {

    public:
    bool fromVar(tk::math::MatIO::var_t &var) { }

    bool toVar(std::string name, tk::math::MatIO::var_t &var) {
        std::string a = "ciao";
        var.set(name, a);
    }
};

int main( int argc, char** argv){

    tk::common::CmdParser cmd(argv, "test matio");
    std::string matfile = cmd.addArg("matfile", "matlab.mat", "path of the mat file");
    std::string varname = cmd.addOpt("-var", "", "var to read");
    int recursive_limit = cmd.addIntOpt("-r", -1, "number of recursive print (-1 infinite)");
    int list_limit = cmd.addIntOpt("-l", -1, "max number of var to print (-1 infinite)");

    cmd.parse();

/*
    tk::math::MatIO mat;
    mat.create(matfile);

    //tk::math::MatIO::var_t var;
    //TestDump t; t.toVar("test", var);
    //mat.write(var);
    //var.release();

    //std::vector<tk::math::MatIO::var_t> structFields(4);
    //int a = 24;
    //double b = 2.5564;
    //Eigen::MatrixXf c; c.resize(2, 2); c << 1, 2, 3, 4;
    //std::string d = "Ciao ciao !!! lol";
    //structFields[0].set("a", a);
    //structFields[1].set("b", b);
    //structFields[2].set("c", c);
    //structFields[3].set("d", d);

    //tk::math::MatIO::var_t structVar;
    //structVar.setStruct("lol", structFields);
    //mat.write(structVar);

    tk::common::Tfpose tf = tk::common::Tfpose::Identity();
    Eigen::MatrixXd matProva; matProva.resize(30, 40);
    tk::math::MatIO::var_t tfVar;
    tfVar.set("tf", tf.matrix());
    tfVar.set("tf", matProva);

    mat.write(tfVar);
    tfVar.release();
    mat.close();
*/
    
    tk::math::MatIO mat;
    mat.open(matfile);
    mat.stats();

    if(varname != "") {
        std::vector<std::string> path = splitString(varname, '.');
        tkASSERT(path.size() >= 1);

        tk::math::MatIO::var_t initial_var, var;
        for(int i=0; i< path.size(); i++) {
            if(i==0) {
                if(!mat.read(path[i], initial_var)) {
                    tkFATAL("var not found")
                }
                var = initial_var;
            } else {
                var = var[path[i]];
            }
        }
        var.print(0, recursive_limit, list_limit);
        initial_var.release();
    } else {
        for(int i=0; i<mat.size(); i++){
            tk::math::MatIO::var_t var;
            mat.read(mat[i], var);
            var.print(0, recursive_limit, list_limit);
            var.release();
        }
    }
    mat.close();


/*
    tk::math::MatIO::var_t var;
    for(int i=0; i<mat.size(); i++) {
        mat.read(mat[i], var);
        var.print();
        var.release();
    }
*/

    /*
    mat.read("lol", structVar);
    structVar.print();
 
    int readA;
    double readB;
    Eigen::MatrixXf readC;
    std::string readD;
    structVar["a"].get(readA);
    structVar["b"].get(readB);
    structVar["c"].get(readC);
    structVar["d"].get(readD);

    std::cout<<readA<<"\n";
    std::cout<<readB<<"\n";
    std::cout<<readC<<"\n";
    std::cout<<readD<<"\n";*/

    /*
    // img2buffer 
    std::vector<uint8_t> buf;
    cv::imencode(".jpg", m, buf);
    // buffer2img
    cv::Mat readM = cv::imdecode(buf, cv::IMREAD_UNCHANGED);
    */
   
    /*
    mat.open(matfile);
    mat.stats();
    tk::math::MatIO::var_t var;
    mat.readVar("key_45", var);
    var.print();
    */
    return 0;
}