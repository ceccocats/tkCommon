#pragma once

#include <tkCommon/math/filter/Tracker.hpp>
#include <tkCommon/math/filter/ConstantVelocityModel.hpp>
#include <tkCommon/math/CSpline2D.h>

namespace tk {
namespace math {

/**
 * @brief Tracker class interface
 */
template <template<class> class F>
class Tracker2D : public Tracker<F, ConstantVelocityModel<double>> {
protected:
    using type = typename Tracker<F, ConstantVelocityModel<double>>::type;
	using VectorN = typename Tracker<F, ConstantVelocityModel<double>>::VectorN;
    using VectorM = typename Tracker<F, ConstantVelocityModel<double>>::VectorM;
    using VectorK = typename Tracker<F, ConstantVelocityModel<double>>::VectorK;
    using MatrixNN = typename Tracker<F, ConstantVelocityModel<double>>::MatrixNN;
    using MatrixKN = typename Tracker<F, ConstantVelocityModel<double>>::MatrixKN;
    using MatrixNM = typename Tracker<F, ConstantVelocityModel<double>>::MatrixNM;
	using MatrixKK = typename Tracker<F, ConstantVelocityModel<double>>::MatrixKK;
	using MatrixNK = typename Tracker<F, ConstantVelocityModel<double>>::MatrixNK;

public:
    using Ptr = std::shared_ptr<Tracker2D>;

    Tracker2D(const uint32_t aID, const timeStamp_t aTimestamp, const VectorN &aInitialState, const MatrixNN &aProcessNoise, const MatrixNN &aCov = MatrixNN::Identity()) 
        : Tracker<F, ConstantVelocityModel<double>>(aID, aTimestamp, aInitialState, aProcessNoise, aCov)
    {}
    ~Tracker2D() = default;

    Eigen::Vector2d position() const {
        Eigen::Vector4d state = this->getState();
        return state.head<2>();
    }

    Eigen::Vector2d velocity() const {
        Eigen::Vector4d state = this->getState();
        return state.tail<2>();
    }

    Eigen::Matrix2d positionCov() const {
        Eigen::Matrix4d covariance = this->getCov();
        return covariance.block<2, 2>(0, 0);
    }

    Eigen::Matrix2d velocityCov() const {
        Eigen::Matrix4d covariance = this->getCov();
        return covariance.block<2, 2>(2, 2);
    }
};
}}