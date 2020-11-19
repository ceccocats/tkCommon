#include <vector>
#include "tkCommon/rt/Lockable.h"

namespace tk { namespace rt {

template<class T>
class Buffer : public tk::rt::Lockable : public std::vector<T>{
};

}}