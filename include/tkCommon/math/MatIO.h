#pragma once
#include "tkCommon/common.h"

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

template <typename Tp> struct matio_type;
template <> struct matio_type<int8_t>   { typedef int8_t type;   static const matio_types tid = MAT_T_INT8; };
template <> struct matio_type<uint8_t>  { typedef uint8_t type;  static const matio_types tid = MAT_T_UINT8; };
template <> struct matio_type<int16_t>  { typedef int16_t type;  static const matio_types tid = MAT_T_INT16; };
template <> struct matio_type<uint16_t> { typedef uint16_t type; static const matio_types tid = MAT_T_UINT16; };
template <> struct matio_type<int32_t>  { typedef int32_t type;  static const matio_types tid = MAT_T_INT32; };
template <> struct matio_type<uint32_t> { typedef uint32_t type; static const matio_types tid = MAT_T_UINT32; };
template <> struct matio_type<float>    { typedef float type;    static const matio_types tid = MAT_T_SINGLE; };
template <> struct matio_type<double>   { typedef double type;   static const matio_types tid = MAT_T_DOUBLE; };
template <> struct matio_type<long double> { typedef double type; static const matio_types tid = MAT_T_DOUBLE; };
template <> struct matio_type<int64_t>  { typedef int64_t type;  static const matio_types tid = MAT_T_INT64; };
template <> struct matio_type<uint64_t> { typedef uint64_t type; static const matio_types tid = MAT_T_UINT64; };

/**
    Mat reader class
*/
class MatIO {

private:
    std::map<std::string, long> fposes; // map var name to position in file
    mat_t *matfp = NULL;

public:

    struct var_t {
        matvar_t *var;
        std::map<std::string, var_t> fields;

        void print(int level = 0) {
            for(int i=0; i<level; i++)
                std::cout<<"|   ";
            std::cout<<var->name<<"\n";
            for(auto f: fields)
                f.second.print(level+1);
        }

        var_t& operator[](std::string s) {
            tkASSERT(fields.count(s) > 0);
            return fields[s];
        }

        bool check(matvar_t *var, matio_types tid, int rank, bool unary = false) {
            if(var == NULL)
                return false;
            if(var->data_type != tid) {
                clsErr("Unable to read value in var " + std::string(var->name) +"\n");
                clsErr("type check failed: " + std::to_string(var->data_type) + " != " + std::to_string(tid) + "\n");
                return false;
            }
            if(var->rank != rank) {
                clsErr("Unable to read value in var " + std::string(var->name) +"\n");
                clsErr("Rank missmatch " + std::to_string(var->rank) + " != " + std::to_string(rank) + "\n");
                return false;
            }
            if(unary && (var->dims[0] != 1 || var->dims[1] != 1) ) {
                clsErr("Unable to read value in var " + std::string(var->name) +"\n");
                clsErr("Dims are not unary: (" + std::to_string(var->dims[0]) + ", " + std::to_string(var->dims[1]) +")\n");
                return false;
            }
        }

        template <class T>
        bool get(T &val) {
            matio_type<T> mat_type;
            if(!check(var, mat_type.tid, 2, true))
                return false;
            memcpy(&val, var->data, var->data_size);
            return true;
        }

        bool get(Eigen::MatrixXf &mat) {
            if(!check(var, MAT_T_SINGLE, 2))
                return false;
            mat.resize(var->dims[0], var->dims[1]);
            memcpy(mat.data(), var->data, mat.size()*var->data_size);
            return true;
        }
        bool get(Eigen::MatrixXd &mat) {
            if(!check(var, MAT_T_DOUBLE, 2))
                return false;
            mat.resize(var->dims[0], var->dims[1]);
            memcpy(mat.data(), var->data, mat.size()*var->data_size);
            return true;
        }

    };

    bool open(std::string mat_dir) {

        matfp = Mat_Open(mat_dir.c_str(), MAT_ACC_RDWR);
        if(matfp == NULL) {
            clsErr("can't open file: " + mat_dir + "\n");
            return false;
        }

        // build fposes map
        while(true) {
            long fpos = ftell((FILE*)matfp->fp);
            matvar_t *var = Mat_VarReadNextInfo(matfp);
            if(var == NULL)
                break;
            fposes[var->name] = fpos;

            Mat_VarFree(var);
        } 

        return true;
    }

    void close() {
        if(matfp != NULL) Mat_Close(matfp);
    }

    void parseMatVar(matvar_t *var, var_t &v) {
        v.var = var;

        // parse struct
        if(var->class_type == MAT_C_STRUCT) {
            int n = Mat_VarGetNumberOfFields(var);
            char* const* names = Mat_VarGetStructFieldnames(var);

            for(int i=0; i<n; i++) {
                matvar_t *ivar = Mat_VarGetStructFieldByIndex(var, i, 0);
                if(ivar != NULL)
                    parseMatVar(ivar, v.fields[names[i]]);
            }
        }

        // parse data
        if(var->class_type >= MAT_C_DOUBLE && var->class_type <= MAT_C_UINT64) {

        }
    }

    bool readVar(std::string key) {
        if(fposes.count(key) == 0)
            return false;
        
        // seek to var position 
        long fpos = fposes[key];
        (void)fseek((FILE*)matfp->fp, fpos, SEEK_SET);

        matvar_t *var = Mat_VarReadNext(matfp);
        if(var == NULL)
            return false;

        var_t v;
        parseMatVar(var, v);
        v.print();
        double id = 0;
        if(v["data"]["gps"]["accX"].get(id))
            std::cout<<id<<"\n";

        Eigen::MatrixXf tf;
        if(v["tf"].get(tf))
            std::cout<<tf<<"\n";     

        Mat_VarFree(var);
        return true;
    }

    void stats() {
        if(matfp == NULL) {
            clsWrn("Matfile not opened\n");
            return;
        }
        clsMsg(std::string(matfp->header) + "\n");
        clsMsg("file: " + std::string(matfp->filename) + "  ( " 
               + std::to_string(fposes.size()) + " vars )\n");
    }

};

}}