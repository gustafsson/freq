#include <QObject>
#include "graphinvalidator.h"

#include <boost/foreach.hpp>

namespace Signal {
namespace Processing {

GraphInvalidator::
        GraphInvalidator(Dag::ptr::weak_ptr dag, INotifier::weak_ptr notifier, Step::ptr::weak_ptr step)
    :
      dag_(dag),
      notifier_(notifier),
      step_(step)
{
}


void GraphInvalidator::
        deprecateCache(Signal::Intervals what) const
{
    Dag::ptr dagp = dag_.lock ();
    if (!dagp)
        return;

    auto dag = dagp.read ();
    INotifier::ptr notifier = notifier_.lock ();
    Step::ptr step = step_.lock ();

    if (!notifier || !step)
        return;

    GraphInvalidator::deprecateCache(*dag, step, what);

    notifier->wakeup();
}


void GraphInvalidator::
        deprecateCache(const Dag& dag, Step::ptr step, Signal::Intervals what)
{
    what = step.write ()->deprecateCache(what);

    for (Step::ptr ts: dag.targetSteps(step)) {
        deprecateCache(dag, ts, what);
    }
}

} // namespace Processing
} // namespace Signal


#include <QThread>
#include "bedroomnotifier.h"

namespace Signal {
namespace Processing {

class WaitForWakeupMock: public QThread {
public:
    WaitForWakeupMock(Bedroom::ptr bedroom) : bedroom_(bedroom) {}

    void run() {
        bedroom_->getBed().sleep();
    }

private:
    Bedroom::ptr bedroom_;
};

void GraphInvalidator::
        test()
{
    // It should invalidate caches and wakeup workers
    {
        // create
        Dag::ptr dag(new Dag);
        Bedroom::ptr bedroom(new Bedroom);
        INotifier::ptr notifier(new BedroomNotifier(bedroom));
        Step::ptr step(new Step(Signal::OperationDesc::ptr()));
        WaitForWakeupMock sleeper(bedroom);

        // wire up
        sleeper.start ();
        dag.write ()->appendStep(step);
        Signal::Intervals initial_valid(-20,60);
        int taskid = step.write ()->registerTask(initial_valid.spannedInterval ());
        (void)taskid; // discard
        EXCEPTION_ASSERT_EQUALS(step.read ()->not_started(), ~initial_valid);
        EXCEPTION_ASSERT(sleeper.isRunning ());

        EXCEPTION_ASSERT_EQUALS(sleeper.wait (1), false);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers (), 1);

        // test
        GraphInvalidator graphInvalidator(dag, notifier, step);
        Signal::Intervals deprected(40,50);
        graphInvalidator.deprecateCache (deprected);

        EXCEPTION_ASSERT_EQUALS(step.read ()->not_started(), ~initial_valid | deprected);
        sleeper.wait (1);
        EXCEPTION_ASSERT_EQUALS(bedroom->sleepers (), 0);
        EXCEPTION_ASSERT(sleeper.isFinished ());
    }
}


} // namespace Processing
} // namespace Signal
