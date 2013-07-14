#ifndef SIGNAL_PROCESSING_GRAPHUPDATER_H
#define SIGNAL_PROCESSING_GRAPHUPDATER_H

#include "dag.h"
#include "invalidator.h"
#include "workerbedroom.h"

namespace Signal {
namespace Processing {

class GraphUpdater: public Invalidator
{
public:
    GraphUpdater(Dag::Ptr dag, WorkerBedroom::Ptr worker_bedroom);

    void deprecateCache(Step::Ptr s, Signal::Intervals what) const;

private:
    // invalidate steps (only deprecateCache(Interval::Interval_ALL) until OperationDesc supports 'affectedInterval' (inverse of requiredInterval))
    void deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const;

    Dag::Ptr dag_;
    WorkerBedroom::Ptr worker_bedroom_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_GRAPHUPDATER_H
