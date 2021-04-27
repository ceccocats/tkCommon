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
            tkPROF_tic(test_time)
            usleep(10000);
            tkPROF_toc(test_time)
            }
            #pragma omp task
            {
            tkPROF_tic(test_time)
            usleep(10000);
            tkPROF_toc(test_time)
            }
            #pragma omp task
            {
            tkPROF_tic(test_time)
            usleep(10000);
            tkPROF_toc(test_time)
            }
        }

        tkPROF_print
    }
}