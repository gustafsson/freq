#ifndef SIGNAL_PROCESSING_TARGETS_H
#define SIGNAL_PROCESSING_TARGETS_H

#include "step.h"
#include "target.h"
#include "targetupdater.h"
#include "workerbedroom.h"
#include "dag.h"

namespace Signal {
namespace Processing {

class Targets: public VolatilePtr<Targets>
{
public:
    Targets(Dag::Ptr dag, WorkerBedroom::Ptr worker_bedroom);

    TargetUpdater::Ptr      addTarget(Step::Ptr step);
    void                    removeTarget(Step::Ptr step);
    std::vector<Step::Ptr>  getTargetSteps() const;
    std::vector<Target::Ptr>  getTargets() const;

private:
    Dag::Ptr dag_;
    WorkerBedroom::Ptr worker_bedroom_;

    typedef std::vector<Target::Ptr> TargetInfos;
    TargetInfos targets;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_TARGETS_H
