#include "chaininfo.h"
#include "signal/processing/workers.h"

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

using namespace Signal::Processing;

namespace Tools {
namespace Support {

ChainInfo::
        ChainInfo(Chain::ConstPtr chain)
    :
      chain_(chain)
{
}


bool ChainInfo::
        hasWork()
{
    return 0 < out_of_date_sum();
}


int ChainInfo::
        n_workers()
{
    Workers::Ptr workers = read1(chain_)->workers();
    return read1(workers)->n_workers();
}


int ChainInfo::
        dead_workers()
{
    Workers::ReadPtr workers (read1(chain_)->workers());
    return workers->workers().size() - workers->n_workers();
}


Signal::UnsignedIntervalType ChainInfo::
        out_of_date_sum()
{
    Signal::Intervals I;

    Targets::Ptr targets = read1(chain_)->targets();
    const std::vector<TargetNeeds::Ptr>& T = read1(targets)->getTargets();
    for (auto i=T.begin(); i!=T.end(); i++)
    {
        const TargetNeeds::Ptr& t = *i;
        I |= read1(t)->out_of_date();
    }

    return I.count ();
}


} // namespace Support
} // namespace Tools

#include "test/operationmockups.h"
#include "test/randombuffer.h"
#include "TaskTimer.h"
#include "log.h"
#include "signal/buffersource.h"
#include <QApplication>
#include <atomic>

namespace Tools {
namespace Support {

class ProcessCrashOperationDesc: public Test::TransparentOperationDesc {
    class exception: virtual public boost::exception, virtual public std::exception {};

    class Operation: public Signal::Operation {
    public:
        Signal::pBuffer process(Signal::pBuffer b) override {
            BOOST_THROW_EXCEPTION(exception() << Backtrace::make ());
        }
    };

    Signal::Operation::Ptr createOperation(Signal::ComputingEngine*) const override {
        return Signal::Operation::Ptr(new Operation);
    }
};

class RequiredIntervalCrash: public Test::TransparentOperationDesc {
    class exception: virtual public boost::exception, virtual public std::exception {};

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* ) const override {
        BOOST_THROW_EXCEPTION(exception() << Backtrace::make ());
    }
};

void ChainInfo::
        test()
{
    std::string name = "ChainInfo";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);

    // It should provide info about the running state of a signal processing chain
    {
        Chain::Ptr cp = Chain::createDefaultChain ();
        ChainInfo c(cp);

        EXCEPTION_ASSERT( !c.hasWork () );
        EXCEPTION_ASSERT_EQUALS( QThread::idealThreadCount () + 1, c.n_workers () );
        EXCEPTION_ASSERT_EQUALS( 0, c.dead_workers () );
    }

    Signal::OperationDesc::Ptr transparent(new Test::TransparentOperationDesc);
    Signal::OperationDesc::Ptr buffersource(new Signal::BufferSource(Test::RandomBuffer::smallBuffer ()));

    // It should say that there is no work if a step has crashed (no crash).
    {
        UNITTEST_STEPS TaskInfo("It should say that there is no work if a step has crashed (no crash)");

        Chain::Ptr cp = Chain::createDefaultChain ();
        ChainInfo c(cp);

        TargetMarker::Ptr at = write1(cp)->addTarget(transparent);
        TargetNeeds::Ptr n = at->target_needs();
        write1(cp)->addOperationAt(buffersource,at);
        EXCEPTION_ASSERT( !c.hasWork () );
        write1(n)->updateNeeds(Signal::Interval(0,10));
        EXCEPTION_ASSERT( c.hasWork () );
        QThread::msleep (10);
        EXCEPTION_ASSERT( !c.hasWork () );
        EXCEPTION_ASSERT_EQUALS( 0, c.dead_workers () );
    }

    // It should say that there is no work if a step has crashed (requiredIntervalCrash).
    {
        UNITTEST_STEPS TaskInfo("It should say that there is no work if a step has crashed (requiredIntervalCrash)");

        Chain::Ptr cp = Chain::createDefaultChain ();
        ChainInfo c(cp);

        TargetMarker::Ptr at = write1(cp)->addTarget(Signal::OperationDesc::Ptr(new RequiredIntervalCrash));
        TargetNeeds::Ptr n = at->target_needs();
        write1(cp)->addOperationAt(buffersource,at);
        EXCEPTION_ASSERT( !c.hasWork () );
        write1(n)->updateNeeds(Signal::Interval(0,10));
        EXCEPTION_ASSERT( c.hasWork () );
        EXCEPTION_ASSERT_EQUALS( 0, c.dead_workers () );
        EXCEPTION_ASSERT( n->sleep (12) );
        EXCEPTION_ASSERT( !c.hasWork () );
        QThread::msleep (1);
        EXCEPTION_ASSERT_EQUALS( 1, c.dead_workers () );
    }

    // It should say that there is no work if a step has crashed (process).
    {
        UNITTEST_STEPS TaskInfo("It should say that there is no work if a step has crashed (process)");

        Chain::Ptr cp = Chain::createDefaultChain ();
        ChainInfo c(cp);

        TargetMarker::Ptr at = write1(cp)->addTarget(Signal::OperationDesc::Ptr(new ProcessCrashOperationDesc));
        TargetNeeds::Ptr n = at->target_needs();
        write1(cp)->addOperationAt(buffersource,at);
        EXCEPTION_ASSERT( !c.hasWork () );
        write1(n)->updateNeeds(Signal::Interval(0,10));
        EXCEPTION_ASSERT( c.hasWork () );
        QThread::msleep (10);
        a.processEvents (); // a crashed worker announces 'wakeup' to the others through the application eventloop
        EXCEPTION_ASSERT( n->sleep (10) );
        EXCEPTION_ASSERT( !c.hasWork () );
        EXCEPTION_ASSERT_EQUALS( 1, c.dead_workers () );
    }
}


} // namespace Support
} // namespace Tools
