#pragma once
#include "tkCommon/math/filter/KalmanFilterBase.hpp"

namespace tk { 
namespace math {
/**
 * @brief			Kalman filter
 * 
 * @param[in] T		type of scalar
 * @param[in] N		dimension of the state
 * @param[in] M		dimension of the input
 * @param[in] K 	dimension of the observation
**/
template <class Model>
class KalmanFilter : public KalmanFilterBase<Model>
{
private:
	using tk::math::KalmanFilterBase<Model>::mX;
	using tk::math::KalmanFilterBase<Model>::mP;
	using tk::math::KalmanFilterBase<Model>::mQ;
	using tk::math::KalmanFilterBase<Model>::mR;
	using tk::math::KalmanFilterBase<Model>::mKG;
	using tk::math::KalmanFilterBase<Model>::mModel;

public:
	using type = typename Model::type;
	using VectorN = typename Model::VectorN;
    using VectorM = typename Model::VectorM;
    using VectorK = typename Model::VectorK;
    using MatrixNN = typename Model::MatrixNN;
    using MatrixKN = typename Model::MatrixKN;
    using MatrixNM = typename Model::MatrixNM;
	using MatrixKK = typename Model::MatrixKK;
	using MatrixNK = typename Model::MatrixNK;

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
	
	/**
	 * @brief Construct a new KalmanFilter object
	 * 
	 * @param aInitialState 		initial state
	 * @param aProcessNoise 		process noise covariance matrix
	 * @param aMeasurementNoise 	measurement noise covariance matrix
	 * @param aCov 					initial covariance matrix
	 */
	KalmanFilter(const VectorN &aInitialState, const MatrixNN &aProcessNoise, const MatrixNN &aCov) 
		: KalmanFilterBase<Model>(aInitialState, aProcessNoise, aCov) {}

	/**
	 * @brief Predict method
	 * 
	 * @param aDt 		delta time
	 * @param aControl 	input
	 */
	void predict(const type aDt, const VectorM &aControl = VectorM::Zero()) override
	{
		mX = mModel.transitionFunction(aDt, mX, aControl);
		mP = mModel.getTransitionMatrix() * mP * mModel.getTransitionMatrix().transpose() + mQ;
	}

	/**
	 * @brief Correct method
	 * 
	 * @param aMeasurement 		measurement vector
	 * @param aMeasurementNoise measurement noise matrix
	 */
	void correct(const VectorK &aMeasurement, const MatrixKK &aMeasurementNoise) override
	{
		mR 	= aMeasurementNoise;
		mKG	= mP * mModel.getMeasurementMatrix().transpose() * (mModel.getMeasurementMatrix() * mP * mModel.getMeasurementMatrix().transpose() + mR).inverse();
		mX 	= mX + mKG * (aMeasurement - mModel.measurementFunction(mX));
		mP 	= (MatrixNN::Identity() - mKG * mModel.getMeasurementMatrix()) * mP;
	}
};
}}