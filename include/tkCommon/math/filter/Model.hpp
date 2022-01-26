#pragma once

#include <Eigen/Dense>

namespace tk {
namespace math {

template <typename T, size_t stateDim, size_t inputDim, size_t measurementDim>
class Model {
public:
    static constexpr size_t sN = stateDim;
    static constexpr size_t sM = inputDim;
    static constexpr size_t sK = measurementDim;
    using type      = T;
    using VectorN   = Eigen::Matrix<T, sN, 1>;
    using VectorM   = Eigen::Matrix<T, sM, 1>;
    using VectorK   = Eigen::Matrix<T, sK, 1>;
    using MatrixNN  = Eigen::Matrix<T, sN, sN>;
    using MatrixKN  = Eigen::Matrix<T, sK, sN>;
    using MatrixNM  = Eigen::Matrix<T, sN, sM>;
    using MatrixKK  = Eigen::Matrix<T, sK, sK>;
	using MatrixNK  = Eigen::Matrix<T, sN, sK>;

     Model() = default;
    ~Model() = default;

    /**
     * @brief 
     * 
     * @param aDt 
     * @param aState 
     * @param aControl 
     * @return VectorN 
     */
    virtual VectorN transitionFunction(T aDt, const VectorN &aState, const VectorM &aControl = VectorM::Zero()) = 0;
    
    /**
     * @brief 
     * 
     * @param aState 
     * @param aControl 
     * @return VectorK 
     */
    virtual VectorK measurementFunction(const VectorN &aState, const VectorM &aControl = VectorM::Zero()) = 0;

    virtual const MatrixNN &getTransitionMatrix() const = 0;
    virtual const MatrixKN &getMeasurementMatrix() const = 0;
    virtual const MatrixNM &getControlMatrix() const = 0;
private:
};
}}