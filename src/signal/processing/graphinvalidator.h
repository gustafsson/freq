#ifndef SIGNAL_PROCESSING_GRAPHUPDATER_H
#define SIGNAL_PROCESSING_GRAPHUPDATER_H

#include "dag.h"
#include "invalidator.h"
#include "bedroom.h"

namespace Signal {
namespace Processing {

class GraphInvalidator: public Invalidator
{
public:
    GraphInvalidator(Dag::Ptr dag, Bedroom::Ptr bedroom);

    void deprecateCache(Step::Ptr s, Signal::Intervals what) const;

private:
    // invalidate steps (only deprecateCache(Interval::Interval_ALL) until OperationDesc supports 'affectedInterval' (inverse of requiredInterval))
    void deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const;

    Dag::Ptr dag_;
    Bedroom::Ptr bedroom_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GRAPHUPDATER_H
