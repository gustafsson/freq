#include "chain.h"
#include "bedroom.h"
#include "firstmissalgorithm.h"
#include "sleepschedule.h"
#include "targetschedule.h"

namespace Signal {
namespace Processing {


Chain::Ptr Chain::
        createDefaultChain()
{
    Dag::Ptr dag(new Dag);
    Bedroom::Ptr bedroom(new Bedroom);
    Targets::Ptr targets(new Targets(dag, bedroom));

    ScheduleAlgorithm::Ptr algorithm(new FirstMissAlgorithm());
    Schedule::Ptr targetSchedule(new TargetSchedule(dag, algorithm, targets));
    Schedule::Ptr sleepSchedule(new SleepSchedule(bedroom, targetSchedule));
    Workers::Ptr workers(new Workers(sleepSchedule));

    for (int i=0; i<QThread::idealThreadCount (); i++) {
        write1(workers)->addComputingEngine(Signal::ComputingEngine::Ptr(new Signal::ComputingCpu));
    }


    Chain::Ptr chain(new Chain(dag, targets, workers));

    return chain;
}


Dag::Ptr Chain::
        dag() const
{
    return dag_;
}


Targets::Ptr Chain::
        targets() const
{
    return targets_;
}


Chain::
        Chain(Dag::Ptr dag, Targets::Ptr targets, Workers::Ptr workers)
    :
      dag_(dag),
      targets_(targets),
      workers_(workers)
{
}


void Chain::
        test()
{
    // It should manage the creation of new signal processing chains
    {
        Chain::Ptr chain = Chain::createDefaultChain ();
        Targets::Ptr targets = read1(chain)->targets();
        Dag::Ptr dag = read1(chain)->dag();

        EXCEPTION_ASSERT(targets);
        EXCEPTION_ASSERT(dag);
    }
}

} // namespace Processing
} // namespace Signal
