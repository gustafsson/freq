#ifndef SIGNAL_COMPUTINGENGINE_H
#define SIGNAL_COMPUTINGENGINE_H

#include <memory>

namespace Signal {

class ComputingEngine
{
public:
    typedef std::shared_ptr<ComputingEngine> ptr;

    virtual ~ComputingEngine();
};


class ComputingCpu: public ComputingEngine {};
class ComputingCuda: public ComputingEngine {};
class ComputingOpenCL: public ComputingEngine {};
class DiscAccessThread: public ComputingEngine {};

} // namespace Signal

#endif // SIGNAL_COMPUTINGENGINE_H
