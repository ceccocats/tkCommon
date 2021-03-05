#pragma once

#include <tkCommon/data/SensorData.h>
#include <vector>

namespace tk{ namespace data {
    
    
    /**
     * @brief  This class implements a vector of omogeneous SensorData
     * 
     */
    template <class T>
    //typename std::enable_if<std::is_base_of<SensorData, T*>::value, void>::type
    class VectorData : public SensorData {    
    public:
        static const DataType type;
        std::vector<T> data;

        /**
         * @brief Get the number of elemnts
         * 
         * @return int size of the vector
         */
        int size() { return data.size(); }

        /**
         * @brief Resize the vector
         * 
         * @param n new element number
         */
        void resize(int n) { data.resize( n ); }

        /**
         * @brief Access to th i element
         * 
         * @param i index
         * @return T& Reference to th i element
         */
        T& operator[] (int i) { return data[i]; }

        /**
         * @brief Initialization method.
         */
        virtual void init() {
            SensorData::init();
        }

        /**
         * @brief Initialization method
         * 
         * @param n number of elements
         */
        virtual void init(int n) {
            SensorData::init();
            data.resize(n);
            for(auto &d : data)
                d.init();
        }

        /**
         * @brief Release method.
         * 
         */
        virtual void release() { 
            for(auto &d : data)
                d.release();
        };

        /**
         * @brief Overloading of operator =
         * Copy all the elements.
         *
         * @param s
         * @return
         */
        VectorData& operator=(const VectorData &s) {
            this->header = s.header;
            data = s.data;
            return *this;
        }

        friend std::ostream& operator<<(std::ostream& os, const VectorData& s){
            for(auto &d : s.data)
                os<<d;
            return os;
        }

        bool toVar(std::string name, tk::math::MatIO::var_t &var) {
            std::vector<tk::math::MatIO::var_t> structVars(2);
            std::vector<tk::math::MatIO::var_t> cellVars( size() );
            SensorData::toVar("header", structVars[0]);
            for(int i = 0; i < size(); i++){
                data[i].toVar("data"+std::to_string(i), cellVars[i]);
            }
            structVars[1].setCells("data", cellVars);

            return var.setStruct(name, structVars);
        }

        bool fromVar(tk::math::MatIO::var_t &var) {

            if(var.empty())
                return false;
            bool ok = true;
            ok = ok && SensorData::fromVar(var["header"]);

            std::vector<tk::math::MatIO::var_t> cellVars;
            // TODO!
            //ok = ok && var["data"].get(cellVars);

            if(!ok) return false;
            
            init(cellVars.size());

            for(int i = 0; i < size(); i++){
                ok = ok && data[i].fromVar(cellVars[i]);
            }

            return ok;
        }

    };

}}