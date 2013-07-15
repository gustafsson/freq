#include <QObject>
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphInvalidator::
        GraphInvalidator(Dag::WeakPtr dag, Bedroom::WeakPtr bedroom, Step::WeakPtr step)
    :
      dag_(dag),
      bedroom_(bedroom),
      step_(step)
{
    EXCEPTION_ASSERT(dag.lock ());
    EXCEPTION_ASSERT(bedroom.lock ());
    EXCEPTION_ASSERT(step.lock ());
}


void GraphInvalidator::
        deprecateCache(Signal::Intervals /*what*/) const
{
    Dag::Ptr dag = dag_.lock ();
    Bedroom::Ptr bedroom = bedroom_.lock ();
    Step::Ptr step = step_.lock ();

    if (!dag || !bedroom || !step)
        return;

    // can't make use of what
    deprecateCache(Dag::ReadPtr(dag), step);

    bedroom->wakeup ();
}


void GraphInvalidator::
        deprecateCache(const Dag::ReadPtr& dag, Step::Ptr s) const
{
    write1(s)->deprecateCache(Signal::Intervals::Intervals_ALL);

    BOOST_FOREACH(Step::Ptr ts, dag->targetSteps(s)) {
        deprecateCache(dag, ts);
    }
}

} // namespace Processing
} // namespace Signal


#include <QThread>

namespace Signal {
namespace Processing {

class WaitForWakeupMock: public QThread {
public:
    WaitForWakeupMock(Bedroom::Ptr bedroom) : bedroom_(bedroom) {}

    void run() {
        bedroom_->sleep();
    }

private:
    Bedroom::Ptr bedroom_;
};

void GraphInvalidator::
        test()
{
    // It should invalidate caches and wakeup workers
    {
        // create
        Dag::Ptr dag(new Dag);
        Bedroom::Ptr bedroom(new Bedroom);
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr()));
        WaitForWakeupMock sleeper(bedroom);

        // wire up
        sleeper.start ();
        write1(dag)->appendStep(step);
        write1(step)->setInvalid(Signal::Intervals(20,30));
        EXCEPTION_ASSERT_EQUALS(read1(step)->not_started(), Signal::Intervals(20,30));
        EXCEPTION_ASSERT(sleeper.isRunning ());

        EXCEPTION_ASSERT_EQUALS(sleeper.wait (1), false);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers (), 1);

        // test
        GraphInvalidator graphInvalidator(dag, bedroom, step);
        Signal::Intervals dummy;
        graphInvalidator.deprecateCache (dummy);

        EXCEPTION_ASSERT_EQUALS(read1(step)->not_started(), Signal::Intervals::Intervals_ALL);
        sleeper.wait (1);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers (), 0);
        EXCEPTION_ASSERT(sleeper.isFinished ());
    }
}


} // namespace Processing
} // namespace Signal
