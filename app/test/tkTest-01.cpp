#include "tkCommon/catch2/catch.hpp"
#include "tkCommon/common.h"
#include "tkCommon/math/Mat.h"
#include "tkCommon/math/MatIO.h"
#include "tkCommon/math/Vec.h"

TEST_CASE("Test mat class") {
    tk::math::Mat<float> mat;
    mat.resize(4, 3);

    SECTION("resizing changes rows, cols and size") {
        REQUIRE(mat.rows() == 4);
        REQUIRE(mat.cols() == 3);
        REQUIRE(mat.size() == (mat.rows() * mat.cols()));
    }

    //SECTION("Accessing out of matrix rise an exception") {
    //    REQUIRE_THROWS(mat(7, 2));
    //    REQUIRE_THROWS(mat(-5, 2));
    //    REQUIRE_THROWS(mat(3, -2222));
    //    REQUIRE_THROWS(mat(2, 66));
    //}

    SECTION("Copy to gpu") {
        mat(1, 2) = 5.0f;
        mat.synchGPU();
        mat.synchCPU();
        REQUIRE(mat(1, 2) == 5.0f);
    }

    SECTION("Fill matrix") {
        float value = 10.0f;
        mat.fill(value);

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == Approx(value));
            }
        }
    }

    SECTION("Operator =") {
        mat.resize(2, 2);
        mat.writableMatrix() << 40.0f, 10.0f, 25.0f, 33.0f;
        tk::math::Mat<float> mat2;
        mat2 = mat;

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == mat2(i, j));
            }
        }
    }

    SECTION("double free") {
        mat.resize(1000, 1000);
        mat.cpu.release();
        mat.cpu.release();
        mat.cpu.release();
        mat.cpu.release();
        mat.cpu.release();
    }


    SECTION("serialize") {
        mat.resize(2, 3);
        mat.writableMatrix() << 0, 1, 2, 3, 4, 5;

        tk::math::MatIO file;
        file.create("test.mat");
        file.write("mat", mat);
        file.close();

        tk::math::Mat<float> mat2;
        file.open("test.mat");
        file.read("mat", mat2);
        file.close();

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == mat2(i, j));
            }
        }  
    }


    SECTION("Eigen ops") {
        mat.resize(3, 3);
        mat.writableMatrix() << 0, 1, 2, 3, 4, 5, 6, 7, 8;

        Eigen::MatrixXf m = mat.matrix();

        mat = mat.matrix()*2;
        m = m*2;

        //std::cout<<m<<"\n";
        //mat.print();

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == m(i, j));
            }
        }  
    }

    SECTION("array of dynamic Mats") {
        mat.resize(3, 3);
        mat.writableMatrix() << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        
        // mat
        {
            std::vector<tk::math::Mat<float>> vec;
            for(int i=0; i<100; i++) {
                vec.resize(i+1);
                vec[i].resize(3,3);
                vec[i].writableMatrix() << 0, 1, 2, 3, 4, 5, 6, 7, 8;
            }
            for(int n=0; n<100; n++) {
                for (int i = 0; i < mat.rows(); i++) 
                for (int j = 0; j < mat.cols(); j++) {
                    REQUIRE(vec[n](i, j) == mat(i, j));
                }
            }
        }

        // mat simple
        //{
        //    std::vector<tk::math::MatSimple<float, false>> vec;
        //    for(int i=0; i<100; i++) {
        //        vec.resize(i+1);
        //        vec[i].resize(3,3);
        //        memcpy(vec[i].data, mat.cpu.data, sizeof(float)*3*3);
        //    }
        //    for(int n=0; n<100; n++) {
        //        for (int i = 0; i < mat.rows(); i++) 
        //        for (int j = 0; j < mat.cols(); j++) {
        //            REQUIRE(vec[n].at(i, j) == mat(i, j));
        //        }
        //    }
        //}
    }


    SECTION("owned") {
        mat.resize(3, 3);
        mat.writableMatrix() << 0, 1, 2, 3, 4, 5, 6, 7, 8;
        tk::math::Mat<float> om(mat.data(), nullptr, 3, 3);
        REQUIRE(om.cpu.owned == true);    
    }

    SECTION("static") {
        std::vector<tk::math::Vec4f> data;            
        for (int i = 0; i<100; i++) {
            tk::math::Vec4f point(1,2,3,4);
            data.push_back(point);
        }
        int sum = 0;
        for(int i=0; i<data.size(); i++) {
            sum += data[i].x();
        }
        REQUIRE(sum == data.size());

        tk::math::Vec<tk::math::Vec2f> data2;
        data2.resize(100);
        for(int i=0; i<data2.size(); i++) {
            data2[i] = tk::math::Vec2f(1,2);
        }
        sum = 0;
        for(int i=0; i<data2.size(); i++) {
            sum += data2[i].x();
        }
        REQUIRE(sum == data2.size());
    }


    SECTION("Satic operator =") {
        tk::math::Mat2f m;
        m.writableMatrix() << 40.0f, 10.0f, 25.0f, 33.0f;
        tk::math::Mat2f m2;
        m2 = m;

        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                REQUIRE(m(i, j) == m2(i, j));
            }
        }
    }

    SECTION("static serialize") {
        tk::math::MatStatic<float,2,3> m;
        m.writableMatrix() << 0, 1, 2, 3, 4, 5;

        tk::math::MatIO file;
        file.create("test.mat");
        file.write("mat", m);
        file.close();
        tk::math::Mat<float> m2;
        file.open("test.mat");
        file.read("mat", m2);
        file.close();
        for (int i = 0; i < m.rows(); i++) {
            for (int j = 0; j < m.cols(); j++) {
                REQUIRE(m(i, j) == m2(i, j));
            }
        }  

        {
            tk::math::Vec<tk::math::MatStatic<float,2,3>> mvec;
            mvec.resize(100);
            for(int i=0; i<mvec.size(); i++) {
                mvec[i] = m;
            }
            file.create("test2.mat");
            file.write("mat", mvec);
            file.close();
            tk::math::Vec<tk::math::MatStatic<float,2,3>> mvec2;
            file.open("test2.mat");
            file.read("mat", mvec2);
            file.close();
            REQUIRE(mvec.size() == mvec2.size());
            for(int n=0; n<mvec.size(); n++) {
                for (int i = 0; i < mvec[n].rows(); i++) {
                    for (int j = 0; j < mvec[n].cols(); j++) {
                        REQUIRE(mvec[n](i, j) == mvec2[n](i, j));
                    }
                }  
            }
        }

        {
            std::vector<tk::math::MatStatic<float,2,3>> mvec;
            for(int i=0; i<mvec.size(); i++) {
                mvec.push_back(m);
            }
            file.create("test3.mat");
            file.write("mat", mvec);
            file.close();
            tk::math::Vec<tk::math::MatStatic<float,2,3>> mvec2;
            file.open("test3.mat");
            file.read("mat", mvec2);
            file.close();
            REQUIRE(mvec.size() == mvec2.size());
            for(int n=0; n<mvec.size(); n++) {
                for (int i = 0; i < mvec[n].rows(); i++) {
                    for (int j = 0; j < mvec[n].cols(); j++) {
                        REQUIRE(mvec[n](i, j) == mvec2[n](i, j));
                    }
                }  
            }
        }
    }

    SECTION("static vector as unic array") {
        // check memory alignment
        REQUIRE(sizeof(tk::math::Vec2<uint8_t>) == sizeof(uint8_t)*2);
        REQUIRE(sizeof(tk::math::Vec3<uint8_t>) == sizeof(uint8_t)*3);
        REQUIRE(sizeof(tk::math::Vec4<uint8_t>) == sizeof(uint8_t)*4);
        REQUIRE(sizeof(tk::math::Vec2<uint16_t>) == sizeof(uint16_t)*2);
        REQUIRE(sizeof(tk::math::Vec3<uint16_t>) == sizeof(uint16_t)*3);
        REQUIRE(sizeof(tk::math::Vec4<uint16_t>) == sizeof(uint16_t)*4);        
        REQUIRE(sizeof(tk::math::Vec2f) == sizeof(float)*2);
        REQUIRE(sizeof(tk::math::Vec3f) == sizeof(float)*3);
        REQUIRE(sizeof(tk::math::Vec4f) == sizeof(float)*4);
        REQUIRE(sizeof(tk::math::Vec2d) == sizeof(double)*2);
        REQUIRE(sizeof(tk::math::Vec3d) == sizeof(double)*3);
        REQUIRE(sizeof(tk::math::Vec4d) == sizeof(double)*4);


        tk::math::Vec<tk::math::Vec3f> mvec;
        mvec.resize(100);
        for(int i=0; i<mvec.size(); i++) {
            mvec[i] = { 0, 1, 2};
        }

        float *ptr = (float*) mvec.data();
        REQUIRE(sizeof(tk::math::Vec3f) == sizeof(float)*3);
        for(int i=0; i<mvec.size()*3; i+=3) {
            REQUIRE(ptr[i+0] == 0);
            REQUIRE(ptr[i+1] == 1);
            REQUIRE(ptr[i+2] == 2);
        }
    }


}