#pragma once
#include "tkCommon/common.h"
#include <tkCommon/terminalFormat.h>

namespace tk { namespace common {

    /**
     *  Cli command parser class
     *  sample usage:
     *      tk::common::CmdParser cmd(argv, "general optional info");
     *      std::string mappath  = cmd.addArg("mappath/", "map/", "path of the map to load");
     *      std::string datapath = cmd.addArg("datapath/", "", "path of the odom data to replay");
     *      bool verbose         = cmd.addBoolOpt("-v", "enable verbose");
     *      std::string confpath = cmd.addOpt("-conf", "conf.yaml", "configuration yaml file");
     *      cmd.print();
     */
    class CmdParser {

    private:
        char **argv_ptr;
        int argv_mode;

        struct Arg {
            std::string name;
            std::string info;
            std::string default_val;
            bool optional;
        };
        struct Opt {
            std::string name;
            std::string info;
            std::string optType;

            std::string default_val_str;
        };

        std::vector<Arg> args;
        std::vector<Opt> opts;
        std::string generalInfo = "";

        const int ARGSW    = 12;
        const int DEFAULTW = 24;

    public:

        /**
         * init with argument array
         * @param argv
         * @param info general info to show at start
         */
        CmdParser(char** argv, std::string info = "");

        /**
         * info to print at program start
         * @param info
         */
        void setGeneralInfo(std::string info);
        /**
         * Add positional argument
         * @param name
         * @param default_val
         * @param info
         * @return
         */
        std::string addArg(std::string name, std::string default_val = "", std::string info = "");

        /**
         * Add bolean option
         * @param opt
         * @param default_val
         * @param info
         * @return
         */
        bool addBoolOpt(std::string opt, std::string info = "");

        /**
         * Add String option
         * @param opt
         * @param default_val
         * @param info
         * @return
         */
        std::string addOpt(std::string opt, std::string default_val = "", std::string info = "");

        /**
         * Add Float option
         * @param opt
         * @param default_val
         * @param info
         * @return
         */
        float addFloatOpt(std::string opt, float default_val = 0, std::string info = "");

        /**
         * Add Integer option
         * @param opt
         * @param default_val
         * @param info
         * @return
         */
        int addIntOpt(std::string opt, int default_val = 0, std::string info = "");


        /**
         * print nice report
         */
        void parse();

        void printUsage(std::string name);
    };




}}
