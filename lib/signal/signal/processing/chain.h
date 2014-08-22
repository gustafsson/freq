#ifndef SIGNAL_PROCESSING_CHAIN_H
#define SIGNAL_PROCESSING_CHAIN_H

#include "shared_state.h"
#include "targets.h"
#include "targetmarker.h"
#include "dag.h"
#include "iinvalidator.h"
#include "inotifier.h"
#include "bedroom.h"

namespace Signal {
namespace Processing {

class Workers;

/**
 * @brief The Chain class should make the signal processing namespace easy to
 * use with a clear and simple interface.
 *
 * It should add signal processing operation steps to the Dag.
 * It should remove steps from the Dag.
 *
 * It should provide means to deprecate caches when the an added operation
 * changes (such as settings or contained data).
 */
class Chain
{
public:
    typedef std::shared_ptr<Chain> ptr;
    typedef std::shared_ptr<const Chain> const_ptr;

    static Chain::ptr createDefaultChain();

    ~Chain();

    /**
     * @brief close prevents any more work from being started on and asks
     * all worker threads to close.
     *
     * Returns true if all threads finished within 'timeout'.
     */
    bool close(int timeout=1000);


    /**
     * @brief addTarget
     * @param desc
     * @param at
     * @return A marker to keep track of the target. The Target is removed from the Dag when TargetMarker is deleted.
     */
    TargetMarker::ptr addTarget(Signal::OperationDesc::ptr desc, TargetMarker::ptr at=TargetMarker::ptr());

    /**
     * @brief addOperation
     * @param desc
     * @param at
     * @return
     *
     * Call IInvalidator::deprecateCache to update the chain with the samples that
     * were affected by this definition. A generic call would be
     * 'invalidator.read ()->deprecateCache(chain->extent(at));'
     */
    IInvalidator::ptr addOperationAt(Signal::OperationDesc::ptr desc, TargetMarker::ptr at);
    void removeOperationsAt(TargetMarker::ptr at);
    Signal::OperationDesc::Extent extent(TargetMarker::ptr at) const;

    shared_state<Workers> workers() const;
    Targets::ptr targets() const;
    shared_state<const Dag> dag() const;

    void resetDefaultWorkers();
    // Add jumping around with targets later.

private:
    Dag::ptr dag_;
    Targets::ptr targets_;
    shared_state<Workers> workers_;
    Bedroom::ptr bedroom_;
    INotifier::ptr notifier_;

    Chain(Dag::ptr, Targets::ptr targets, shared_state<Workers> workers, Bedroom::ptr bedroom, INotifier::ptr notifier);

    Step::ptr::weak_ptr createBranchStep (Dag& dag, Signal::OperationDesc::ptr desc, TargetMarker::ptr at);
    Step::ptr::weak_ptr insertStep (Dag& dag, Signal::OperationDesc::ptr desc, TargetMarker::ptr at);

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_CHAIN_H
