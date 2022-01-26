#pragma once

#include <tkCommon/math/filter/KalmanFilterBase.hpp>

namespace tk {
namespace math {

/**
 * @brief Tracker class interface
 */
template <template<class> class F, class M>
class Tracker {
protected:
    using type = typename F<M>::type;
	using VectorN = typename F<M>::VectorN;
    using VectorM = typename F<M>::VectorM;
    using VectorK = typename F<M>::VectorK;
    using MatrixNN = typename F<M>::MatrixNN;
    using MatrixKN = typename F<M>::MatrixKN;
    using MatrixNM = typename F<M>::MatrixNM;
	using MatrixKK = typename F<M>::MatrixKK;
	using MatrixNK = typename F<M>::MatrixNK;

    uint32_t mID;
    uint32_t mCorrectionCount;
    timeStamp_t mInitTime;               // time when the tracker was initialized
    timeStamp_t mLastPredictionTime;     // tiem when prediction was performed
    timeStamp_t mLastCorrectionTimer;    // time when correction was performed
    std::unique_ptr<F<M>> mFilter;
public:    
    /**
     * @brief constructor
     * @param id            tracker ID
     * @param time          timestamp
     * @param init_pos      initial position
     * @param associated    associated detection
     */
    Tracker(const uint32_t aID, const timeStamp_t aTimestamp, const VectorN &aInitialState, const MatrixNN &aProcessNoise, const MatrixNN &aCov = MatrixNN::Identity())
        : mID(aID), mCorrectionCount(0), mInitTime(aTimestamp), mLastPredictionTime(aTimestamp), mLastCorrectionTimer(aTimestamp)
    {
        mFilter.reset(new F<M>(aInitialState, aProcessNoise, aCov));
    }
    ~Tracker() {}
    using Ptr = std::shared_ptr<Tracker>;

  
public:
    /**
     * @brief predict the current state
     * @param time    current time
     */
    virtual void predict(const timeStamp_t aTimestamp, const VectorM &aControl = VectorM::Zero()) {
        type dt = type(aTimestamp - mLastPredictionTime) * type(1e-6);
        dt = std::max((type) 0.001, dt);
        
        mFilter->predict(dt, aControl);

        mLastPredictionTime = aTimestamp;
    }

    /**
     * @brief correct the state with an observation
     * @param time    current time
     * @param pos     observed position
     * @param associated   associated detection
     */
    virtual void correct(const timeStamp_t aTimestamp, const VectorK& aMeasurement, const MatrixKK &aMeasurementNoise) {
        mFilter->correct(aMeasurement, aMeasurementNoise);
        mCorrectionCount++;
        mLastCorrectionTimer = aTimestamp;
    }

    uint32_t id() const {
        return mID;
    }

    timeStamp_t age(const timeStamp_t aTimestamp) const {
        return (aTimestamp - mInitTime);
    }

    timeStamp_t lastCorrectionTime() const {
        return mLastCorrectionTimer;
    }

    const VectorN &getState() const {
        return mFilter->getState();
    }

    const MatrixNN &getCov() const {
        return mFilter->getCov();
    }

    uint32_t correctionCount() const {
        return mCorrectionCount;
    }

    void printInternalState() const {
        mFilter->printInternalState();
    }
};

}}
