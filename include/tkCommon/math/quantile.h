//Here is Quantile function which is MATLAB's equivalent with linear interpolation:
// https://stackoverflow.com/a/37708864

    
#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace tk { namespace math {

template<typename T>
static inline double lerp(T v0, T v1, T t) {
    return (1 - t)*v0 + t*v1;
}

template<typename T>
static inline std::vector<T> quantile(const std::vector<T>& inData, const std::vector<T>& probs) {
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

}}
