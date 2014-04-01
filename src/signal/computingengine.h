#ifndef SIGNAL_COMPUTINGENGINE_H
#define SIGNAL_COMPUTINGENGINE_H

#include <boost/weak_ptr.hpp>

namespace Signal {

class ComputingEngine
{
public:
    typedef std::shared_ptr<ComputingEngine> Ptr;

    virtual ~ComputingEngine();
};


class ComputingCpu: public ComputingEngine {};
class ComputingCuda: public ComputingEngine {};
class ComputingOpenCL: public ComputingEngine {};

} // namespace Signal

#endif // SIGNAL_COMPUTINGENGINE_H
