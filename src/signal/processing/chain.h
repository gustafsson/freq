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
 * @brief The Chain class should make the signal processing namespace easy to
 * use with a clear and simple interface.
 *
 * Doesn't really need VolatilePtr since all member variables are thread safe
 * by themselves. But using VolatilePtr makes it more clear that this class is
 * indeed thread-safe. A bit far fetched maybe, ah well.
 *
 * TODO
 * ----
 * extent should return an OperationDesc::Extent.
 */
class Chain: public VolatilePtr<Chain>
{
public:
    static Chain::Ptr createDefaultChain();

    ~Chain();

    TargetNeeds::Ptr addTarget(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at=TargetNeeds::Ptr());
    /**
     * @brief addOperation
     * @param desc
     * @param at
     * @return
     *
     * Call IInvalidator::deprecateCache to update the chain with the samples that
     * were affected by this definition. A generic call would be
     * 'read1(invalidator)->deprecateCache(chain->extent(at));'
     */
    IInvalidator::Ptr addOperationAt(Signal::OperationDesc::Ptr desc, TargetNeeds::Ptr at);
    void removeOperationsAt(TargetNeeds::Ptr at);
    Signal::Interval extent(TargetNeeds::Ptr at) const;

    void print_dead_workers() const;
    void rethrow_worker_exception() const;

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
