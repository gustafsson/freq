#ifndef SIGNAL_PROCESSING_GRAPHUPDATER_H
#define SIGNAL_PROCESSING_GRAPHUPDATER_H

#include "dag.h"
#include "iinvalidator.h"
#include "inotifier.h"

namespace Signal {
namespace Processing {

/**
 * @brief The GraphInvalidator class should invalidate caches and wakeup workers.
 *
 * It will silently stop doing anything if any of it's dependencies are deleted.
 */
class GraphInvalidator: public IInvalidator
{
public:
    GraphInvalidator(Dag::WeakPtr dag, INotifier::WeakPtr notifier, Step::WeakPtr step);

    void deprecateCache(Signal::Intervals what) const;
    static void deprecateCache(const Dag& dag, Step::Ptr s, Signal::Intervals what);

private:

    Dag::WeakPtr dag_;
    INotifier::WeakPtr notifier_;
    Step::WeakPtr step_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GRAPHUPDATER_H
