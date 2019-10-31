#pragma once

#include <map>
#include <string>

namespace tk { namespace tcolor{



    namespace foreground{

        const static char   reset[]             = "39";
        const static char   black[]             = "30";
        const static char   red[]               = "31";
        const static char   green[]             = "32";
        const static char   yellow[]            = "33";
        const static char   blue[]              = "34";
        const static char   magenta[]           = "35";
        const static char   cyan[]              = "36";
        const static char   lightGray[]         = "37";
        const static char   darkGray[]          = "90";
        const static char   lightRed[]          = "91";
        const static char   lightGreen[]        = "92";
        const static char   lightYellow[]       = "93";
        const static char   lightBlue[]         = "94";
        const static char   lightMagenta[]      = "95";
        const static char   lightCyan[]         = "96";
        const static char   white[]             = "97";

    }

    namespace background{

        const static char   reset[]             = "49";
        const static char   black[]             = "40";
        const static char   red[]               = "41";
        const static char   green[]             = "42";
        const static char   yellow[]            = "43";
        const static char   blue[]              = "44";
        const static char   magenta[]           = "45";
        const static char   cyan[]              = "46";
        const static char   lightGray[]         = "47";
        const static char   darkGray[]          = "100";
        const static char   lightRed[]          = "101";
        const static char   lightGreen[]        = "102";
        const static char   lightYellow[]       = "103";
        const static char   lightBlue[]         = "104";
        const static char   lightMagenta[]      = "105";
        const static char   lightCyan[]         = "106";
        const static char   white[]             = "107";

    }


    /**
     * @brief                       Color formatting text
     * 
     * @param source                String to formatting
     * @param foreground_color      Foreground color
     * @param background_color      Background color
     * @return std::string          String Formatted
     */
    static std::string 
    print(std::string const& str, std::string foreground_color = "", std::string background_color = "")
    {
        std::string const start = "\033[0;";
        std::string const end   = "\033[0m";

        if(foreground_color == ""){
            foreground_color = foreground::reset;
        }

        if(background_color == ""){
            background_color = background::reset;
        }

        return start + foreground_color + std::string{";"} + background_color + std::string{"m"} + str + end;
    }

}}
