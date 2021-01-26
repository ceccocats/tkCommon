#include "tkCommon/CmdParser.h"
#include "tkCommon/term_utils.h"
#include "tkCommon/math/Mat.h"
#include <thread>
#include <signal.h>

__global__
void print_cuda(tk::math::Mat_t<float> m)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Mat dims: %d %d\n", m.rows, m.cols);
  m.data[i] = i;
}

int main( int argc, char** argv){
    tk::common::CmdParser cmd(argv, "test eigen with cuda");
    cmd.parse();

    tk::math::Mat<float> m;
    m.useGPU();
    m.resize(3,3); 
    print_cuda<<<1, 9>>>(m.gpu);
    cudaDeviceSynchronize();
    m.synchCPU();
    m.print();

    return 0;
}
