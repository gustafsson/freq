#ifndef SIGNAL_COMPUTINGENGINE_H
#define SIGNAL_COMPUTINGENGINE_H

namespace Signal {

class ComputingEngine
{
public:
    virtual ~ComputingEngine();
};


class ComputingCpu {};
class ComputingCuda {};
class ComputingOpenCL {};

} // namespace Signal

#endif // SIGNAL_COMPUTINGENGINE_H
