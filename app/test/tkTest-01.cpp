#include "tkCommon/catch2/catch.hpp"
#include "tkCommon/math/Mat.h"
#include "tkCommon/math/MatIO.h"

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
        mat << 40.0f, 10.0f, 25.0f, 33.0f;
        tk::math::Mat<float> mat2;
        mat2 = mat;

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == mat2(i, j));
            }
        }
    }


    SECTION("serialize") {
        mat.resize(2, 3);
        mat << 0, 1, 2, 3, 4, 5;

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
        mat << 0, 1, 2, 3, 4, 5, 6, 7, 8;

        Eigen::MatrixXf m = mat.matrix();

        mat.matrix() = mat.matrix()*2;
        m = m*2;

        //std::cout<<m<<"\n";
        //mat.print();

        for (int i = 0; i < mat.rows(); i++) {
            for (int j = 0; j < mat.cols(); j++) {
                REQUIRE(mat(i, j) == m(i, j));
            }
        }  
    }
}