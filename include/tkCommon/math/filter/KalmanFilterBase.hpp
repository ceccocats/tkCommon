#pragma once

namespace tk {
namespace math {
/**
 * @brief			Generic Kalman filter
 * 
 * @param[in] T		type of scalar
 * @param[in] N		dimension of the state
 * @param[in] M		dimension of the input
 * @param[in] K 	dimension of the observation
**/
template <class Model>
class KalmanFilterBase {
protected:
	using type = typename Model::type;
	using VectorN = typename Model::VectorN;
    using VectorM = typename Model::VectorM;
    using VectorK = typename Model::VectorK;
    using MatrixNN = typename Model::MatrixNN;
    using MatrixKN = typename Model::MatrixKN;
    using MatrixNM = typename Model::MatrixNM;
	using MatrixKK = typename Model::MatrixKK;
	using MatrixNK = typename Model::MatrixNK;

	VectorN mX;		// state
	MatrixNN mP;	// filter covariance
	MatrixNN mQ;	// process noise cov
	MatrixKK mR;	// mesurement noise cov
	MatrixNK mKG;	// filter gain
	Model mModel;
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	/**
	 * @brief Construct a new KalmanFilterBase object
	 * 
	 * @param aInitialState 		initial state
	 * @param aProcessNoise 		process noise covariance matrix
	 * @param aMeasurementNoise 	measurement noise covariance matrix
	 * @param aCov 					initial covariance matrix
	 */
	KalmanFilterBase(const VectorN &aInitialState, const MatrixNN &aProcessNoise, const MatrixNN &aCov){
		mX = aInitialState;
		mP = aCov;
		mQ = aProcessNoise;
	}

	/**
	 * @brief Predict method
	 * 
	 * @param aDt 		delta time
	 * @param aControl 	input
	 */
	virtual void predict(const type aDt, const VectorM &aControl = VectorM::Zero()) = 0;

	/**
	 * @brief Correct method
	 * 
	 * @param aMeasurement 		measurement vector
	 * @param aMeasurementNoise measurement noise matrix
	 */
	virtual void correct(const VectorK &aMeasurement, const MatrixKK &aMeasurementNoise) = 0;
	
	const VectorN  &getState() const { return mX; }
	const MatrixNN &getCov() const { return mP; }
	const MatrixNN &getTransitionMatrix() const { return mModel.getTransitionMatrix(); }
	const MatrixNM &getControlMatrix() const { return mModel.getControlMatrix(); }
	const MatrixKN &getMeasurementMatrix() const { return mModel.getMeasurementMatrix(); }
	const MatrixNN &getProcessNoiseCov() const { return mQ; }
	const MatrixKK &getMeasurementNoiseCov() const { return mR; }
	const MatrixNK &getKalmanGain() const { return mKG; }
	void setMean(const VectorN &aMean) { mX = aMean; }
	void setCov(const MatrixNN &aCov) { mP = aCov; }
	void setProcessNoiseCov(const MatrixNN &aProcessNoiseCov) { mQ = aProcessNoiseCov; }
	void setMeasurementNoiseCov(const MatrixKK &aMeasurementNoiseCov) { mR = aMeasurementNoiseCov; }

	void printInternalState() const {
		std::cout << "***************Internal state*********************" << std::endl;
		std::cout << "n_states: " << mX.size() << std::endl;
		std::cout << "state: " << mX.transpose().matrix() << std::endl;
		std::cout << "P: \n"
				<< mP.matrix() << std::endl;
		std::cout << "R: \n"
				<< mR.matrix() << std::endl;
		std::cout << "Q: \n"
				<< mQ.matrix() << std::endl;
		std::cout << "H: \n"
				<< mModel.getMeasurementMatrix().matrix() << std::endl;
		std::cout << "K: \n"
				<< mKG.matrix() << std::endl;
		std::cout << "A: \n"
				<< mModel.getTransitionMatrix().matrix() << std::endl;
		std::cout << "B: \n"
				<< mModel.getControlMatrix().matrix() << std::endl;
		std::cout << "**************************************************" << std::endl;
	}
};
}}