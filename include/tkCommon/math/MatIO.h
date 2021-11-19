#pragma once
#include <utility>
#include <type_traits>
#include <matio.h>
#include "tkCommon/common.h"
#include "tkCommon/math/MatSimple.h"

/** @if mat_devman
 * @brief Matlab MAT File information
 *
 * Contains information about a Matlab MAT file
 * @ingroup mat_internal
 * @endif
 */
struct _mat_t {
    void  *fp;              /**< File pointer for the MAT file */
    char  *header;          /**< MAT file header string */
    char  *subsys_offset;   /**< Offset */
    char  *filename;        /**< Filename of the MAT file */
    int    version;         /**< MAT file version */
    int    byteswap;        /**< 1 if byte swapping is required, 0 otherwise */
    int    mode;            /**< Access mode */
    long   bof;             /**< Beginning of file not including any header */
    size_t next_index;      /**< Index/File position of next variable to read */
    size_t num_datasets;    /**< Number of datasets in the file */
#if defined(MAT73) && MAT73
    hid_t  refs_id;         /**< Id of the /#refs# group in HDF5 */
#endif
    char **dir;             /**< Names of the datasets in the file */
};

namespace tk { namespace  math {

template <typename Tp> struct matio_type {
    static const matio_types tid = MAT_T_UNKNOWN;   
    static const matio_classes cid = MAT_C_EMPTY;   
};
template <> struct matio_type<int8_t>   { typedef int8_t type;    static const matio_types tid = MAT_T_INT8;   static const matio_classes cid = MAT_C_INT8;   };
template <> struct matio_type<uint8_t>  { typedef uint8_t type;   static const matio_types tid = MAT_T_UINT8;  static const matio_classes cid = MAT_C_UINT8;  };
template <> struct matio_type<int16_t>  { typedef int16_t type;   static const matio_types tid = MAT_T_INT16;  static const matio_classes cid = MAT_C_INT16;  };
template <> struct matio_type<uint16_t> { typedef uint16_t type;  static const matio_types tid = MAT_T_UINT16; static const matio_classes cid = MAT_C_UINT16; };
template <> struct matio_type<int32_t>  { typedef int32_t type;   static const matio_types tid = MAT_T_INT32;  static const matio_classes cid = MAT_C_INT32;  };
template <> struct matio_type<uint32_t> { typedef uint32_t type;  static const matio_types tid = MAT_T_UINT32; static const matio_classes cid = MAT_C_UINT32; };
template <> struct matio_type<int64_t>  { typedef int64_t type;   static const matio_types tid = MAT_T_INT64;  static const matio_classes cid = MAT_C_INT64;  };
template <> struct matio_type<uint64_t> { typedef uint64_t type;  static const matio_types tid = MAT_T_UINT64; static const matio_classes cid = MAT_C_UINT64; };
template <> struct matio_type<float>    { typedef float type;     static const matio_types tid = MAT_T_SINGLE; static const matio_classes cid = MAT_C_SINGLE; };
template <> struct matio_type<double>   { typedef double type;    static const matio_types tid = MAT_T_DOUBLE; static const matio_classes cid = MAT_C_DOUBLE; };
template <> struct matio_type<long double> { typedef double type; static const matio_types tid = MAT_T_DOUBLE; static const matio_classes cid = MAT_C_DOUBLE; };

class MatDump;

/**
    Mat file reader class
*/
class MatIO {

private:
    // WARNING: fposes and fposesMas should always be updated togheder
    std::vector<std::string> fposes;       // list of vars in file
    std::map<std::string, long> fposesMap; // map var name to position in file
    mat_t *matfp = NULL;

public:

    struct var_t {
    private:
        matvar_t *var = NULL;  // matio raw data pointer

        // WARNING: fieldMap and fields should always be updated togheder
        std::vector<std::string> fields;        // list of fieldMap keys (for easy iteration)
        std::map<std::string, var_t> fieldMap;  // key -> var mapping

        bool check(matvar_t *var, matio_types tid, int rank, bool unary = false) {
            if(var == NULL) {
                tkERR("unable to read var is NULL");
                return false;
            }
            if(var->data == NULL) {
                tkERR("unable to read var->data is NULL");
                return false;
            }
            if(var->data_type != tid) {
                tkERR("Unable to read value in var " + std::string(var->name) +"\n");
                tkERR("type check failed: " + std::to_string(var->data_type) + " != " + std::to_string(tid) + "\n");
                return false;
            }
            if(var->rank != rank) {
                tkERR("Unable to read value in var " + std::string(var->name) +"\n");
                tkERR("Rank missmatch " + std::to_string(var->rank) + " != " + std::to_string(rank) + "\n");
                return false;
            }
            if(unary && (var->dims[0] != 1 || var->dims[1] != 1) ) {
                tkERR("Unable to read value in var " + std::string(var->name) +"\n");
                tkERR("Dims are not unary: (" + std::to_string(var->dims[0]) + ", " + std::to_string(var->dims[1]) +")\n");
                return false;
            }
            return true;
        }

    public:

        void print(int level = 0, int recursive_limit = -1, int list_limit = -1) {
            if(var == NULL)
                return;
            if(var->name == NULL)
                return;

            for(int i=0; i<level; i++)
                std::cout<<"|   ";
            std::cout<<var->name;

            if(var->class_type == MAT_C_STRUCT) {
                std::cout<<":";
            } else if(var->class_type == MAT_C_CELL) {
                std::cout<<" {"<<var->dims[0]<<", "<<var->dims[1]<<"}";

            } else if(var->dims[0] != 1 || var->dims[1] != 1) {
                std::cout<<" (";
                for(int i=0; i<var->rank; i++) {
                    std::cout<<var->dims[i];
                    if(i < var->rank-1)
                        std::cout<<", ";
                }
                std::cout<<")";
            }
            std::cout<<"\n";

            if(recursive_limit-- == 0) 
                return;

            int limit = 0;
            for(auto f: fieldMap) {
                if(list_limit >= 0 && ++limit > list_limit) {
                    std::cout<<"...\n";
                    break;
                }
                f.second.print(level+1, recursive_limit, list_limit);
            } 
        }

        /**
         * @brief recursively parse raw matvar_t
         */
        void parse(matvar_t *var) {
            release();

            this->var = var;

            // parse struct
            if(var->class_type == MAT_C_STRUCT) {
                int n = Mat_VarGetNumberOfFields(var);
                char* const* names = Mat_VarGetStructFieldnames(var);

                for(int i=0; i<n; i++) {
                    matvar_t *ivar = Mat_VarGetStructFieldByIndex(var, i, 0);
                    if(ivar != NULL) {
                        std::string key = names[i];
                        fieldMap[key].parse(ivar);
                        fields.push_back(key);
                    }
                }
                tkASSERT(this->fieldMap.size() == this->fields.size());
            }
            // parse cell
            else if(var->class_type == MAT_C_CELL) {
                int n = var->dims[0]*var->dims[1];
                for(int i=0; i<n; i++) {
                    matvar_t *ivar = Mat_VarGetCell(var, i);
                    if(ivar != NULL) {
                        if(ivar->name == NULL)
                            ivar->name = strdup( std::to_string(i).c_str() );

                        std::string key = std::to_string(i);
                        fieldMap[key].parse(ivar);
                        fields.push_back(key);
                    }
                }
                tkASSERT(this->fieldMap.size() == this->fields.size());
            }       
        }

        /**
         * @brief true if no data
         */
        bool empty() {
            return var == NULL;
        }

        /**
         * @brief deallocate data and clear structure
         */
        void release() {
            if(var != NULL) 
                Mat_VarFree(var);
            var = NULL;
            fieldMap.clear();
            fields.clear();
        }

        /**
         * @brief Raw matvar_t pointer
         */
        matvar_t* getRawData() {
            return var;
        }

        /**
         * @brief this var name
         */
        std::string name() {
            if(var == NULL)
                return "null";
            return var->name;
        }

        /**
         * @brief Number of fields. works with structure and cells array
         */
        int size() {
            tkASSERT(fieldMap.size() == fields.size());
            return fieldMap.size();
        }

        /**
         * @brief iterate fields name
         */
        std::string operator[](int i) {
            tkASSERT(i < fields.size());
            return fields[i];
        }

        /**
         * @brief get field var
         * @param s field name
         */
        var_t& operator[](std::string s) {
            tkASSERT(fieldMap.count(s) > 0);
            return fieldMap[s];
        }


        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(T &a);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(T &a){
            matio_type<T> mat_type;
            if(!check(var, mat_type.tid, 2, true))
                return false;
            memcpy(&a, var->data, var->data_size);
            return true;
        }
        template<typename T, int A, int B>
        bool get(Eigen::Matrix<T, A, B> &mat) {
            matio_type<T> mat_type;
            if(!check(var, mat_type.tid, 2))
                return false;
            mat.resize(var->dims[0], var->dims[1]);
            memcpy(mat.data(), var->data, mat.size()*var->data_size);
            return true;
        }
        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(MatSimple<T,false> &mat);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(MatSimple<T,false> &mat) {
            matio_type<T> mat_type;
            if(!check(var, mat_type.tid, 2))
                return false;
            mat.resize(var->dims[0], var->dims[1]);
            memcpy(mat.data, var->data, mat.size*var->data_size);
            return true;
        }
        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(std::vector<T> &vec);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool get(std::vector<T> &vec) {
            matio_type<T> mat_type;
            if(!check(var, mat_type.tid, 2))
                return false;
            vec.resize(var->dims[0]*var->dims[1]);
            memcpy(vec.data(), var->data, vec.size()*var->data_size);
            return true;
        }
        bool get(std::string &vec) {
            matio_type<uint8_t> mat_type;
            if(!check(var, mat_type.tid, 2))
                return false;
            vec.resize(var->dims[0]*var->dims[1]);
            memcpy( (void*)vec.data(), var->data, vec.size()*var->data_size);
            return true;
        }
        template<typename T>
        bool get(tk::common::Map<T> &map) {
            for(int i=0; i<fields.size(); i++) {
                map.add(fields[i]);
                fieldMap[fields[i]].get(map[fields[i]]);
            }
            return true;
        }


        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, T &a);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, T &a) {
            matio_type<T> mat_type;
            if(mat_type.tid == MAT_T_UNKNOWN) {
                tkERR("could not serialize this type\n");
                return false;
            }
            release();
            size_t dim[2] = { 1, 1 }; // 1x1, single value
            var = Mat_VarCreate(name.c_str(), mat_type.cid, mat_type.tid, 2, dim, &a, 0);
            return true;
        } 
        template<typename T, int A, int B>
        bool set(std::string name, Eigen::Matrix<T, A, B> &mat) {
            matio_type<T> mat_type;
            if(mat_type.tid == MAT_T_UNKNOWN) {
                tkERR("could not serialize this type\n");
                return false;
            }
            release();
            size_t dim[2] = { (size_t) mat.rows(), (size_t) mat.cols() }; 
            var = Mat_VarCreate(name.c_str(), mat_type.cid, mat_type.tid, 2, dim, mat.data(), 0);
            return true;
        }
        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, MatSimple<T,false> &mat);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, MatSimple<T,false> &mat) {
            matio_type<T> mat_type;
            if(mat_type.tid == MAT_T_UNKNOWN) {
                tkERR("could not serialize this type\n");
                return false;
            }
            release();
            size_t dim[2] = { (size_t) mat.rows, (size_t) mat.cols }; 
            var = Mat_VarCreate(name.c_str(), mat_type.cid, mat_type.tid, 2, dim, mat.data, 0);
            return true;
        }
        template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, std::vector<T> &vec);
        template<class T, typename std::enable_if<!std::is_base_of<tk::math::MatDump, T>::value, int>::type = 0>
        bool set(std::string name, std::vector<T> &vec) {
            matio_type<T> mat_type;
            if(mat_type.tid == MAT_T_UNKNOWN) {
                tkERR("could not serialize this type\n");
                return false;
            }
            release();
            size_t dim[2] = { (size_t) vec.size(), 1 }; 
            var = Mat_VarCreate(name.c_str(), mat_type.cid, mat_type.tid, 2, dim, vec.data(), 0);
            return true;
        }
        bool set(std::string name, std::string &vec) {
            release();
            size_t dim[2] = { 1, (size_t) vec.size() }; 
            var = Mat_VarCreate(name.c_str(), MAT_C_CHAR, MAT_T_INT8, 2, dim, (void*)vec.data(), 0);
            return true;
        }
        template<typename T>
        bool set(std::string name, tk::common::Map<T> &map) {
            std::vector<tk::math::MatIO::var_t> vars(map.size());
            for(int i=0; i<map.size(); i++) {
                vars[i].set(map.keys()[i], *map.vals()[i]);
            }
            return setStruct(name, vars);
        }

        /**
         * @brief Set this var as a Struct 
         * @param name var name
         * @param fields vector of fields as var_t
         */
        bool setStruct(std::string name, std::vector<var_t> fields) {
            release();
            const char *names[fields.size()];
            for(int i=0; i<fields.size(); i++) {
                if(fields[i].var == NULL) {
                    tkERR("provided NULL field while creating struct: " + name  + "\n");
                    return false;
                }
                names[i] = fields[i].var->name;
            }
            size_t dim[2] = { 1, 1 }; // create 1x1 struct
            var = Mat_VarCreateStruct(name.c_str(), 2, dim, names, fields.size()); //main struct: Data
            if(var == NULL)
                return false;

            for(int i=0; i<fields.size(); i++) {
                Mat_VarSetStructFieldByName(var, fields[i].var->name, 0, fields[i].var); //0 for first row
                std::string key = std::string(fields[i].var->name);
                this->fieldMap[key] = fields[i];
                this->fields.push_back(key);
                //std::cout<<"key: "<<key<<"\n";
            }

            //std::cout<<"field missmatch: "<<this->fieldMap.size()<<" "<<this->fields.size()<<"\n";
            tkASSERT(this->fieldMap.size() == this->fields.size());
            return true;
        }

        /**
         * @brief Set this var as a CellArray 
         * @param name var name
         * @param fields vector of fields as var_t
         */
        bool setCells(std::string name, std::vector<var_t> fields) {
            release();
            matvar_t *cells[fields.size()];
            for(int i=0; i<fields.size(); i++) {
                if(fields[i].var == NULL) {
                    tkERR("provided NULL field while creating struct: " + name  + "\n");
                    return false;
                }
                cells[i] = fields[i].var;
            }
            size_t dim[2] = { fields.size(), 1 }; // create 1x1 struct
            var = Mat_VarCreate(name.c_str(),MAT_C_CELL,MAT_T_CELL,2,dim,cells,0);
            if(var == NULL)
                return false;
                
            for(int i=0; i<fields.size(); i++) {
                std::string key = std::to_string(i);
                this->fieldMap[key] = fields[i];
                this->fields.push_back(key);
            }
            tkASSERT(this->fieldMap.size() == this->fields.size());
            return true;
        }
    };

    /**
     * @brief Create new mat file. Override existing one.
     * @param mat_dir path of the mat file
     */
    bool create(std::string mat_dir) {
        close();
        
        matfp = Mat_CreateVer(mat_dir.c_str(), NULL, MAT_FT_MAT5);
        if(matfp == NULL) {
            tkERR("can't create file: " + mat_dir + "\n");
            return false;
        }
        return true;
    }

    /**
     * @brief Open a mat file
     * @param mat_dir path of the mat file
     */
    bool open(std::string mat_dir) {
        close();

        matfp = Mat_Open(mat_dir.c_str(), MAT_ACC_RDWR);
        if(matfp == NULL) {
            tkERR("can't open file: " + mat_dir + "\n");
            return false;
        }

        // build fposesMap map
        while(true) {
            long fpos = ftell((FILE*)matfp->fp);
            matvar_t *var = Mat_VarReadNextInfo(matfp);
            if(var == NULL)
                break;

            std::string key = var->name; 
            fposes.push_back(key);
            fposesMap[key] = fpos;

            Mat_VarFree(var);
        } 

        tkASSERT(fposes.size() == fposesMap.size());
        return true;
    }

    /**
     * @brief Close matfile
     */
    void close() {
        if(matfp != NULL) Mat_Close(matfp);
        matfp = NULL;
        fposes.clear();
        fposesMap.clear();        
    }

    /**
     * @brief Print mat statistics
     */
    void stats() {
        if(matfp == NULL) {
            tkWRN("Matfile not opened\n");
            return;
        }
        tkASSERT(fposes.size() == fposesMap.size());

        tkMSG(std::string(matfp->header) + "\n");
        tkMSG("file: " + std::string(matfp->filename) + "  ( " 
               + std::to_string(fposesMap.size()) + " vars )\n");
    }

    /**
     * @brief get number of vars in file
     */
    int size() {
        tkASSERT(fposes.size() == fposesMap.size());
        return fposesMap.size();
    }

    /**
     * @brief Iterate vars names 
     */
    std::string operator[](int i) {
        tkASSERT(i < fposes.size());
        return fposes[i];
    }

    /**
     * @brief Read var from file
     * @param key var name
     * @param v ouput filled var_t
     */
    bool read(std::string key, var_t &v) {
        if(matfp == NULL)
            return false;

        if(fposesMap.count(key) == 0)
            return false;
        
        // seek to var position 
        long fpos = fposesMap[key];
        (void)fseek((FILE*)matfp->fp, fpos, SEEK_SET);

        matvar_t *var = Mat_VarReadNext(matfp);
        if(var == NULL)
            return false;

        v.parse(var);
        return true;
    }

    /**
     * @brief write var to file
     * @param var input var to write
     */
    bool write(var_t &var) {
        if(matfp == NULL)
            return false;
        if(var.empty())
            return false;
        Mat_VarWrite(matfp, var.getRawData(), MAT_COMPRESSION_NONE);
        return true;
    }

    template<typename T>
    bool read(std::string key, T &v) {
        var_t var;
        bool ok  = read(key, var);
        if(ok) {
            ok = var.get(v);
        }
        var.release();
        return ok;
    }

    template<typename T>
    bool write(std::string key, T &v) {
        var_t var;
        var.set(key, v);
        bool ok = write(var);
        var.release();
        return ok;
    }

};

/**
 * @brief Interface to extends in class
 */
class MatDump {
    public: 

    virtual bool toVar(std::string name, MatIO::var_t &var) {
        tkFATAL("Not implemented");
        return false;
    }
    virtual bool fromVar(MatIO::var_t &var) {
        tkFATAL("Not implemented");
        return false;
    }

    bool loadMat(std::string file, std::string name = "data") {
        tk::math::MatIO mat;
        if(mat.open(file)) {
            tk::math::MatIO::var_t var;
            mat.read(name, var);
            bool ok = fromVar(var);
            var.release();
            mat.close();
            return ok;
        }
        return false;
    }

    bool saveMat(std::string file, std::string name = "data") {
        tk::math::MatIO mat;
        if(mat.create(file)) {
            tk::math::MatIO::var_t var;
            bool ok = toVar(name, var);
            ok = ok && mat.write(var);
            std::string version_str = tk::common::tkVersionGit();
            ok = ok && mat.write("tkversion", version_str);
            std::string date_str = getTimeStampString();
            ok = ok && mat.write("date", date_str);
            var.release();
            mat.close();
            return ok;
        }
        return false;
    }

};

// template specialization
template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::get(T &m) {
    return m.fromVar(*this);
}
template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::get(MatSimple<T,false> &mat) {
    bool ok = true;
    mat.resize(size(), 1);
    for(int i = 0; i < size(); i++){
        ok = ok && mat.data[i].fromVar((*this)[(*this)[i]]);
    }
    return ok;
}
template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::get(std::vector<T> &vec) {
    bool ok = true;
    vec.resize(size());
    for(int i = 0; i < size(); i++){
        ok = ok && vec[i].fromVar((*this)[(*this)[i]]);
    }
    return ok;
}

template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::set(std::string name, T &m) {
    return m.toVar(name, *this);
}
template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::set(std::string name, MatSimple<T,false> &mat) {
    std::vector<tk::math::MatIO::var_t> vars(mat.size);
    for(int i=0; i<mat.size; i++) {
        mat.data[i].toVar(name, vars[i]);
    }
    return setCells(name, vars);
}
template<class T, typename std::enable_if<std::is_base_of<tk::math::MatDump, T>::value, int>::type>
bool MatIO::var_t::set(std::string name, std::vector<T> &vec) {
    std::vector<tk::math::MatIO::var_t> vars(vec.size());
    for(int i=0; i<vec.size(); i++) {
        vec[i].toVar(name, vars[i]);
    }
    return setCells(name, vars);
}


}}