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
class TrackerSD : public Tracker<F, ConstantVelocityModel<double>> {
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

    tk::math::CSpline2D *mSpline;
public:
    TrackerSD(const uint32_t aID, const timeStamp_t aTimestamp, tk::math::CSpline2D *aSpline, const VectorN &aInitialState, const MatrixNN &aProcessNoise, const MatrixNN &aCov = MatrixNN::Identity()) 
        //: Tracker(aID, aTimestamp, aInitialState, aProcessNoise, aCov), mSpline(aSpline) 
    {

        VectorN state;
        //state.head<2>() = cart2frenet(aInitPos);
        //state.tail<2>() = Eigen::Vector2d::Zero();
        this->mID = 0;
        this->mFilter.reset(new F<ConstantVelocityModel<double>>(state, aProcessNoise, aCov));
    }
    ~TrackerSD() {}

    void correct(const timeStamp_t aTimestamp, const VectorK& aMeasurement, const MatrixKK &aMeasurementNoise) override
    {
        /*
        auto sd = cart2frenet(aPos);
        auto mean = mFilter->getState();

        // Forcing the s to be bound in the interval [0, L]
        double track_length = mSpline->s_end;
        double cur_s = mean[0];
        while (cur_s >= track_length) cur_s -= track_length;
        while (cur_s < 0.0f) cur_s += track_length;
        mean[0] = cur_s;
        mFilter->setMean(mean);

        // Updating the s_new to ensure that the error along s computed inside
        // the Kalaman filter is in the interval [-L/2, L/2]
        auto s_error = sd.x() - cur_s;
        s_error = s_error - std::ceil(s_error/track_length - 0.5f) * track_length;
        sd.x() = cur_s + s_error;

        // apply correction
        mFilter->correct(sd, aMeasurementCov);
        */
        this->mCorrectionCount++;
        this->mLastCorrectionTimer = aTimestamp;
    }
private:
    Eigen::Vector2d cart2frenet(const Eigen::Vector2d &aPose, double aGuess = 0.0) const {
        // project
        tk::common::Vector2<float> proj = mSpline->calc_projection(tk::common::Vector2<float>{aPose.x(), aPose.y()}, aGuess);

        tk::common::Vector2<float> a = mSpline->calc_position(proj.x() - 1.0);
        tk::common::Vector2<float> b = mSpline->calc_position(proj.x() + 1.0);
        tk::common::Vector2<float> c(aPose.x(), aPose.y());
        if (tk::common::pointIsleft(a, b, c))
            proj.y() = -proj.y();
        
        return Eigen::Vector2d(proj.x(), proj.y());
    }
};

}}