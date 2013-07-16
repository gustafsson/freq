#ifndef SIGNAL_PROCESSING_CHAIN_H
#define SIGNAL_PROCESSING_CHAIN_H

#include "volatileptr.h"
#include "targets.h"
#include "dag.h"
#include "workers.h"
#include "graphinvalidator.h"

namespace Signal {
namespace Processing {

/**
 * @brief The Chain class should manage the creation of new signal processing chains.
 *
 * It should provide the public interface for managing a signal processing chain.
 *
 * Doesn't really need VolatilePtr since all member variables are thread safe
 * by themselves. But using VolatilePtr
 */
class Chain: public VolatilePtr<Chain>
{
public:
    static Chain::Ptr createDefaultChain();

    TargetNeeds::Ptr addTarget(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at);
    IInvalidator::Ptr addOperation(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at);
    void removeOperations(TargetNeeds::Ptr at);
    Signal::Interval extent(TargetNeeds::Ptr at) const;

    // Add jumping around with targets later.

private:
    Dag::Ptr dag_;
    Targets::Ptr targets_;
    Workers::Ptr workers_;
    Bedroom::Ptr bedroom_;

    Chain(Dag::Ptr, Targets::Ptr targets, Workers::Ptr workers, Bedroom::Ptr bedroom);

    Step::Ptr insertStep(const Dag::WritePtr& dag, Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at);

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_CHAIN_H
