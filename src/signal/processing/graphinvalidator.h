#ifndef SIGNAL_PROCESSING_GRAPHUPDATER_H
#define SIGNAL_PROCESSING_GRAPHUPDATER_H

#include "dag.h"
#include "iinvalidator.h"
#include "bedroom.h"

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
    GraphInvalidator(Dag::WeakPtr dag, Bedroom::WeakPtr bedroom, Step::WeakPtr step);

    void deprecateCache(Signal::Intervals what) const;

private:
    void deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s, Signal::Intervals what) const;

    Dag::WeakPtr dag_;
    Bedroom::WeakPtr bedroom_;
    Step::WeakPtr step_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GRAPHUPDATER_H
