#include "tkCommon/CmdParser.h"
#include "tkCommon/term_utils.h"
#include "tkCommon/math/Mat.h"
#include "tkCommon/math/Vec.h"
#include <thread>
#include <signal.h>

__global__
void print_cuda(tk::math::MatSimple<float, true> m)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  printf("Mat dims: %d %d, i: %d, val: %f\n", m.rows, m.cols, i, m.data[i]);
  m.data[i] = i;
}

int main( int argc, char** argv){
    tk::common::CmdParser cmd(argv, "test eigen with cuda");
    cmd.parse();

    tk::math::Mat<float> m;
    m.useGPU();
    m.resize(3,3); 
    m.writableMatrix() << 1, 0, 0,
                            0, 1, 0,
                            0, 0, 1;
    m.print();
    tk::math::Mat<float> m2;
    m2 = m;
    m2.print();
    m2(1,1) = 10;
    m2.print();
    m.print();
    

    print_cuda<<<1, 9>>>(m.gpu);
    cudaDeviceSynchronize();
    m.synchCPU();
    m.print();

    m = m.matrix() * m.matrix().transpose();
    m.print();

    std::cout<<"Static mats\n";
    tk::math::Mat4f mat4, mat4_2;
    mat4 = Eigen::MatrixXf::Identity(4,4)*2;
    mat4_2 = Eigen::MatrixXf::Identity(4,4)*4;
    mat4.print();
    mat4_2.print();
    mat4 = mat4.matrix() * mat4_2.matrix();
    mat4.print();

    // wrt matrix
    tk::math::Mat3d cov;
    cov.writableMatrix() << 10, 0, 0,
                            0, 4, 0,
                            0, 0, 5;
    cov.print();


    // owned matrix
    std::cout<<"OWNED\n";
    float data[6] = { 2, 3, 4, 5, 6, 7 };
    tk::math::Mat<float> om(data, nullptr, 3, 2);
    std::cout<<"IS OWNED: "<<om.cpu.owned<<"\n";
    om.print();
    om.synchGPU();
    print_cuda<<<1, 1>>>(om.gpu);
    cudaDeviceSynchronize();
    om.synchCPU();
    om.print();
    std::cout<<"IS OWNED: "<<om.cpu.owned<<"\n";
    return 0;
}
