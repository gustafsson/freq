#include <QObject>
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphInvalidator::
        GraphInvalidator(Dag::Ptr dag, Bedroom::Ptr bedroom)
    :
      dag_(dag),
      bedroom_(bedroom)
{
    EXCEPTION_ASSERT(dag);
    EXCEPTION_ASSERT(bedroom_);
}


void GraphInvalidator::
        deprecateCache(Step::Ptr s, Signal::Intervals /*what*/) const
{
    deprecateCache(Dag::ReadPtr(dag_), s);

    bedroom_->wakeup ();
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
        Step::Ptr step(new Step(Signal::OperationDesc::Ptr(), 1, 2));
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
        GraphInvalidator graphInvalidator(dag, bedroom);
        Signal::Intervals dummy;
        graphInvalidator.deprecateCache (step, dummy);

        EXCEPTION_ASSERT_EQUALS(read1(step)->not_started(), Signal::Intervals::Intervals_ALL);
        sleeper.wait (1);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers (), 0);
        EXCEPTION_ASSERT(sleeper.isFinished ());
    }
}


} // namespace Processing
} // namespace Signal
