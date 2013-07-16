#include "scheduler.h"

#include <QMutexLocker>

using namespace boost;
using namespace std;

namespace Signal {
namespace Dag {


void Scheduler::
        addDagHead(DagHead::Ptr p) volatile
{
    WritePtr self(this);
    self->dag_heads_.insert (p);
    self->dag_head_i_ = self->dag_heads_.begin ();

    self->connect(DagHead::WritePtr(p), SIGNAL(invalidatedSamples()), self, SLOT(invalidatedSamples()));
    self->invalidatedSamples ();
}


void Scheduler::
        removeDagHead(DagHead::Ptr p) volatile
{
    WritePtr self(this);
    self->dag_heads_.erase (p);
    self->dag_head_i_ = self->dag_heads_.begin ();

    self->disconnect(DagHead::WritePtr(p), SIGNAL(invalidatedSamples()), self, SLOT(invalidatedSamples()));
    self->invalidatedSamples ();
}


void Scheduler::
        invalidatedSamples()
{
    QMutexLocker l( &invalidated_samples_mutex_ );
    invalidated_samples_wait_.wakeAll ();
}


void Scheduler::
        sleepUntilWork() volatile
{
    {
        ReadPtr self(this);
        for (std::set<DagHead::Ptr>::iterator i=self->dag_heads_.begin ();
             i != self->dag_heads_.end ();
             ++i)
        {
            DagHead::ReadPtr dag(*i);
            if (dag->invalidSamples())
            {
                // There are invalid samples somewhere so don't sleep. Our round robin will find them.
                return;
            }
        }
    }
    {
        // cast away volatile to access objects which are inherently thread-safe
        Scheduler* self = const_cast<Scheduler*> (this);
        QMutexLocker l( &self->invalidated_samples_mutex_ );

        // Can't use ReadPtr/WritePtr(this) because they would keep a lock during this call.
        self->invalidated_samples_wait_.wait ( &self->invalidated_samples_mutex_ );
    }
}


void Scheduler::
        run(ComputingEngine*e) volatile
{
    while (true)
    {
        Task t = getNextTask (e);
        t = run (t, e);

        sleepUntilWork();
    }
}

Scheduler::Task Scheduler::
        run(const Task& t, ComputingEngine* e) volatile
{
    if (!t.node)
        return Task();

    if (!t.node->startSampleProcessing(t.expected_result))
        return Task();

    Signal::Operation::Ptr o = Node::WritePtr (t.node)->data()->operation (e);
    EXCEPTION_ASSERTX ( o, "Expected getNextTask to return an operation that supportes computing engine e" );

    // Do computing without keeping any lock.
    Signal::pBuffer result = o->process (t.data);
    EXCEPTION_ASSERT_EQUALS (t.expected_result, result->getInterval ());

    t.node->validateSamples( result );

    return t;
}


Scheduler::Task Scheduler::
        getNextTask(ComputingEngine*e) volatile
{
    // Only checks the current dag after rotating it.

    Signal::Intervals missing;
    Node::Ptr node;

    {
        WritePtr self(this);
        // Check if list of dags is empty
        if (self->dag_head_i_ == self->dag_heads_.end ())
        {
            // Return an empty task
            return Task();
        }

        // round-robin between tasks
        self->dag_head_i_++;
        if (self->dag_head_i_ == self->dag_heads_.end ())
            self->dag_head_i_ = self->dag_heads_.begin ();

        DagHead::ReadPtr p(*self->dag_head_i_);
        missing = p->invalidSamples ();
        node = p->head ();
    }

    if (missing) {
        Task t = searchjob(e, node, missing);
        if (t.node)
            return t;
    }

    // Return an empty task
    return Task();
}


Scheduler::Task Scheduler::
        searchjob(ComputingEngine* engine, Node::Ptr node, const Signal::Intervals& missing)
{
    // We need everything in 'missing' from 'node'. Ask the node what we can start with.
    // Don't bother to check the cache, assume someone already did (because if the caller did it would already have found a task on the parent).
    Signal::Operation::Ptr operation;
    Signal::Intervals stillmissing;

    {
        Node::WritePtr nodew(node);
        operation = nodew->data()->operation (engine);
        stillmissing = missing - nodew->data()->current_processing;

        if (0 == nodew->numChildren()) {
            if (operation) {
                // fs and number of channels are just dummy placeholders to compute which interval to fetch.
                Interval I = missing.fetchFirstInterval ();
                return Task(node, Signal::pBuffer (new Signal::Buffer (I, 1, 1)), I);
            } else {
                // Operation doesn't support this engine. Search somewhere else.
                return Task();
            }
        }
    }

    Node::Ptr childp = node->getChild();

    while (stillmissing) {
        Signal::Interval expectedOutput;
        Signal::Intervals requiredInterval = Node::ReadPtr(node)->data()->operationDesc().requiredInterval (stillmissing, &expectedOutput);
        stillmissing -= expectedOutput;

        {
            // See if requiredInterval is availble in the childrens caches. Keep a separate scope from the searchjob call.
            Node::ReadPtr child(childp);
            const Cache& childcache = child->data()->cache;

            if (childcache.samplesDesc ().contains (requiredInterval)) {
                // Nice, the child contains the data we need to get started.
                // But does the child even support this engine?
                if (!operation) {
                    // See if there's any need to dig further (if something that's need is not found in the child cache).
                    // Otherwise this engine is out of job under this node with 'missing'.
                    continue;
                }

                // perform some work right away (Cache::read)...
                // If we just save the interval we can't be sure that
                // childcache won't be invalidated before we get to read its
                // contents.
                pBuffer buffer = childcache.read (requiredInterval.fetchFirstInterval ());
                return Task (node, buffer, expectedOutput);
            }

            requiredInterval -= childcache.samplesDesc ();
        }

        // requiredInterval is not available and desc says that it is the most important thing to provide for stillmissing.
        // see if the child can find us a job
        Task t = searchjob(engine, childp, requiredInterval);
        if (t.data)
            return t;

        // child couldn't provide us with anything meaningful either. Keep on searching until everything has been checked.
    }

    // No job found here
    return Task();
}


} // namespace Dag
} // namespace Signal


namespace Signal {
namespace Dag {

class DummyHeadKernel:public Operation {
public:
    DummyHeadKernel(DagHead::Ptr daghead):daghead(daghead) {}

    virtual Signal::pBuffer process(Signal::pBuffer b) {
        Signal::Interval I = Signal::Intervals(b->getInterval ()).shrink (5).spannedInterval ();
        Signal::pBuffer r(new Signal::Buffer(I, b->sample_rate (), b->number_of_channels ()));
        *r |= *b;

        DagHead::WritePtr head(daghead);
        head->setInvalidSamples(head->invalidSamples() - I);
        return r;
    }

private:
    DagHead::Ptr daghead;
};


class DummyHeadDesc:public OperationDesc {
public:
    DummyHeadDesc(DagHead::Ptr daghead):daghead(daghead) {}

    Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const {
        Interval i = I;
        i.first = i.first / 5 * 5;
        i.last = i.first + 5;
        if (expectedOutput)
            *expectedOutput = i;
        return Intervals (i).enlarge (5).spannedInterval ();
    }

    Signal::Interval affectedInterval (const Interval &i) const {
        return Intervals (i).enlarge (5+4).spannedInterval ();
    }

    virtual OperationDesc::Ptr copy() const { return OperationDesc::Ptr (new DummyHeadDesc(daghead)); }

    virtual Operation::Ptr createOperation(ComputingEngine*) const {
        return Operation::Ptr (new DummyHeadKernel (daghead));
    }

private:
    DagHead::Ptr daghead;
};


void Scheduler::
        test()
{
    OperationDesc::Ptr desc (new BufferSource (pBuffer (new Buffer (Interval (10, 20), 1, 1))));
    Scheduler::Ptr scheduler (new Scheduler ());
    DagHead::Ptr head(new DagHead (SignalDag::Ptr(), desc));
    scheduler->addDagHead( head );
    ComputingEngine* engine = 0;
    Task t = scheduler->getNextTask( engine );
    EXCEPTION_ASSERT_EQUALS( t.data, Signal::pBuffer() );
    EXCEPTION_ASSERT_EQUALS( t.expected_result, Signal::Interval() );
    EXCEPTION_ASSERT_EQUALS( t.node, Node::Ptr() );

    write1(head)->setInvalidSamples( Interval (15, 25));
    t = scheduler->getNextTask( engine );

    EXCEPTION_ASSERT( t.data );
    EXCEPTION_ASSERT( t.node );
    EXCEPTION_ASSERT_EQUALS( t.expected_result, Interval(15,25));
    EXCEPTION_ASSERT_EQUALS( t.data->getInterval (), Interval(15,25));
    EXCEPTION_ASSERT_EQUALS( read1(t.node)->name().toStdString(), "Signal::BufferSource");

    scheduler->run(t, 0);

    // Running shouldn't affect head
    Task t2 = scheduler->getNextTask( engine );
    EXCEPTION_ASSERT( t2.data );
    EXCEPTION_ASSERT( t2.node );
    EXCEPTION_ASSERT_EQUALS( t.expected_result, t2.expected_result );
    EXCEPTION_ASSERT( *t.data == *t2.data );
    EXCEPTION_ASSERT( t.node == t2.node );

    OperationDesc::Ptr dummyDesc = OperationDesc::Ptr(new DummyHeadDesc(head) );
    ICommand::Ptr command(new CommandAddUnaryOperation( dummyDesc ));
    write1(head)->queueCommand( command );

    Node::Ptr node = read1(head)->head();
    EXCEPTION_ASSERT_EQUALS( read1(node)->data()->operationDesc(), *desc );

    write1(head)->executeQueue();

    node = read1(head)->head();
    EXCEPTION_ASSERT_EQUALS( read1(node)->data()->operationDesc(), *dummyDesc );

    t = scheduler->getNextTask( engine );

    EXCEPTION_ASSERT_EQUALS( t.expected_result, Interval(10,25));
    EXCEPTION_ASSERT_EQUALS( t.data->getInterval (), Interval(10,25));
    EXCEPTION_ASSERT_EQUALS( read1(t.node)->name().toStdString(), "Signal::BufferSource");

    scheduler->run(t, 0);

    t = scheduler->getNextTask( engine );

    EXCEPTION_ASSERT_EQUALS( t.expected_result, Interval(15,20));
    EXCEPTION_ASSERT_EQUALS( t.data->getInterval (), Interval(10,25));
    EXCEPTION_ASSERT_EQUALS( read1(t.node)->name().toStdString(), "Signal::Dag::DummyHeadDesc");

    scheduler->run(t, 0);

    t = scheduler->getNextTask( engine );

    EXCEPTION_ASSERT_EQUALS( t.expected_result, Interval(25,30));
    EXCEPTION_ASSERT_EQUALS( t.data->getInterval (), Interval(25,30));
    EXCEPTION_ASSERT_EQUALS( read1(t.node)->name().toStdString(), "Signal::BufferSource");

    scheduler->run(t, 0);

    t = scheduler->getNextTask( engine );

    EXCEPTION_ASSERT_EQUALS( t.expected_result, Interval(20,25));
    EXCEPTION_ASSERT_EQUALS( t.data->getInterval (), Interval(15,30));
    EXCEPTION_ASSERT_EQUALS( read1(t.node)->name().toStdString(), "Signal::Dag::DummyHeadDesc");

    scheduler->run(t, 0);

    EXCEPTION_ASSERT( read1(head)->invalidSamples().empty() );
}


} // namespace Dag
} // namespace Signal
