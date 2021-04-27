#include "tkCommon/catch2/catch.hpp"
#include "tkCommon/common.h"
#include "tkCommon/math/Mat.h"
#include "tkCommon/math/MatIO.h"
#include "tkCommon/math/Vec.h"
#include <cuda.h>
#include "tkCommon/rt/Profiler.h"

TEST_CASE("Test tprof") {
  
    SECTION("Print"){
        #pragma omp parallel
        #pragma omp single
        for(int i=0; i<100; i++) {
            #pragma omp task
            {
            tkPROF_tic(lol)
            usleep(10000);
            tkPROF_toc(lol)
            }
            #pragma omp task
            {
            tkPROF_tic(lol)
            usleep(10000);
            tkPROF_toc(lol)
            }
            #pragma omp task
            {
            tkPROF_tic(lol)
            usleep(10000);
            tkPROF_toc(lol)
            }
        }

        tkPROF_print
    }
}