#pragma once
#include "tkCommon/data/HeaderData.h"
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace data {

    enum class_type : uint8_t{
        UINT8  = 0,
        UINT16 = 1,
        INT8   = 2,
        INT16  = 3,
        FLOAT  = 4,
        STRING = 5
    };    

    template <typename Ta> struct T_to_class_type;
    template <> struct T_to_class_type<uint8_t>     { static const class_type id = UINT8;   void print(std::ostream& os = std::cout){ os<<"uint8_t";  }; };
    template <> struct T_to_class_type<uint16_t>    { static const class_type id = UINT16;  void print(std::ostream& os = std::cout){ os<<"uint16_t"; }; };
    template <> struct T_to_class_type<int8_t>      { static const class_type id = INT8;    void print(std::ostream& os = std::cout){ os<<"int8_t";   }; };
    template <> struct T_to_class_type<int16_t>     { static const class_type id = INT16;   void print(std::ostream& os = std::cout){ os<<"int16_t";  }; };
    template <> struct T_to_class_type<float>       { static const class_type id = FLOAT;   void print(std::ostream& os = std::cout){ os<<"float";  }; };
    template <> struct T_to_class_type<std::string> { static const class_type id = STRING;  void print(std::ostream& os = std::cout){ os<<"string"; }; };


    /**
     * @brief Basic data class.
     * This class is a basic data class that just contains basic information that all sensor data class must contain.
     * @see HeaderData
     */
	class SensorData : public tk::math::MatDump, public tk::rt::Lockable{
    public:
        HeaderData  header;                 /**< Header, @see HeaderData */

        /**
         * @brief Initialization method.
         * Must be implemented by child classes, and will handle the allocation of member variables, if any.
         */
        virtual void init() {
            header.init();
        }

        /**
         * @brief Release method.
         * Must be implemented by child classes, and will handle the deallocation of member variables, if any,
         */
        virtual void release() { tkERR("release method not implemented"); tkFATAL("abort"); };

        /**
         * @brief Overloading of operator =
         * Copy only the header.
         *
         * @param s
         * @return
         */
        SensorData& operator=(const SensorData &s) {
            this->header = s.header;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const SensorData& s){
            os<<"header.stamp: "<<s.header.stamp;
            return os;
        }

        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            return header.toVar(name, var);
        }

        bool fromVar(tk::math::MatIO::var_t &var) {
            return header.fromVar(var);
        }
    };
}}
