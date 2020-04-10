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

/**
    Mat reader class
*/
class MatIO {

private:
    std::map<std::string, long> fposes; // map var name to position in file
    mat_t *matfp = NULL;

public:

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

    bool readVar(std::string key) {
        if(fposes.count(key) == 0)
            return false;
        
        // seek to var position 
        long fpos = fposes[key];
        (void)fseek((FILE*)matfp->fp, fpos, SEEK_SET);

        matvar_t *var = Mat_VarReadNext(matfp);
        std::cout<<"VAR: "<<var<<" "<<var->name<<"\n";
        Mat_VarFree(var);
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