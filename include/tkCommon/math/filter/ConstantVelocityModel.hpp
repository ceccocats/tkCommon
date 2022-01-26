#pragma once

#include <tkCommon/math/filter/Model.hpp>

namespace tk {
namespace math {

template <typename T>
class ConstantVelocityModel : public Model<T, 4, 2, 2>
{
public:
    using VectorN = typename Model<T, 4, 2, 2>::VectorN;
    using VectorM = typename Model<T, 4, 2, 2>::VectorM;
    using VectorK = typename Model<T, 4, 2, 2>::VectorK;
    using MatrixNN = typename Model<T, 4, 2, 2>::MatrixNN;
    using MatrixKN = typename Model<T, 4, 2, 2>::MatrixKN;
    using MatrixNM = typename Model<T, 4, 2, 2>::MatrixNM;

private:
    MatrixNN mA;	// transition matrix
    MatrixKN mH;	// mesurement matrix
    MatrixNM mB;	// control matrix
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    ConstantVelocityModel() {
        mA = MatrixNN::Identity();
        mH = MatrixKN::Identity();
        mB = MatrixNM::Identity();
    }
    ~ConstantVelocityModel() = default;

    VectorN transitionFunction(T aDt, const VectorN &aState, const VectorM &aControl = VectorM::Zero()) 
    {
        mA(0, 2) = aDt;
        mA(1, 3) = aDt;
        return mA * aState + mB * aControl;
    }

    VectorK measurementFunction(const VectorN &aState, const VectorM &aControl = VectorM::Zero())
    {
        return mH * aState;
    }

    const MatrixNN &getTransitionMatrix() const
    {
        return mA;
    }

    const MatrixKN &getMeasurementMatrix() const
    {
        return mH;
    }

    const MatrixNM &getControlMatrix() const
    {
        return mB;
    }

    static Eigen::Vector<T, 2> position(const VectorN &aState) {
        return aState.head(2);
    }

    static Eigen::Vector<T, 2> velocity(const VectorN &aState) {
        return aState.tail(2);
    }
    
    static Eigen::Matrix<T, 2, 2> positionCov(const MatrixNN &aCov) {
        return aCov.block<2, 2>(0, 0);
    }

    static Eigen::Matrix<T, 2, 2> velocityCov(const MatrixNN &aCov) {
        return aCov.block<2, 2>(2, 2);
    }
};
    
}
}