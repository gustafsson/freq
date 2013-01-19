#include "scheduler.h"

#include <QMutexLocker>

namespace Signal {
namespace Dag {


void Scheduler::
        addDagHead(DagHead::Ptr p) volatile
{
    WritePtr self(this);
    self->dag_heads_.insert (p);
    self->dag_head_i_ = self->dag_heads_.begin ();

    self->connect(DagHead::WritePtr(p), SIGNAL(invalidatedSamples()), self, SLOT(invalidatedSamples()));
}


void Scheduler::
        removeDagHead(DagHead::Ptr p) volatile
{
    WritePtr self(this);
    self->dag_heads_.erase (p);
    self->dag_head_i_ = self->dag_heads_.begin ();

    self->disconnect(DagHead::WritePtr(p), SIGNAL(invalidatedSamples()), self, SLOT(invalidatedSamples()));
}


void Scheduler::
        invalidatedSamples()
{
    QMutexLocker l( &task_lock_ );
    task_wait_.wakeOne ();
}


void Scheduler::
        sleepUntilWork() volatile
{
    LOG_ERROR("Not implemented");

    // This logic is flawed. self will be locked while waiting for a signal on task_lock.
    WritePtr self(this);
    QMutexLocker l( &self->task_lock_ );
    self->task_wait_.wait ( &self->task_lock_ );
}


Scheduler::Task Scheduler::
        getNextTask(ComputingEngine* engine) volatile
{
    WritePtr self(this);
    return self->getNextTask(engine);
}


Scheduler::Task Scheduler::
        getNextTask(ComputingEngine*e)
{
    // Check if list of dags is empty
    if (dag_head_i_ == dag_heads_.end ())
    {
        // Return an empty task
        return Task();
    }

    // round robin between tasks
    dag_head_i_++;
    if (dag_head_i_ == dag_heads_.end ())
        dag_head_i_ = dag_heads_.begin ();

    //DagHead::Ptr p = *dag_head_i_;
    Signal::Intervals I;
    Node::Ptr node;
    {
        DagHead::ReadPtr p(*dag_head_i_);
        I = p->invalidSamples ();
        node = p->head ();
    }

    Node::ReadPtr rnode(node);
    const Node::NodeData* data = rnode->data();
    const OperationDesc& desc = data->operationDesc ();
    Signal::Interval expectedOutput;
    Signal::Interval requiredInterval = desc.requiredInterval (I, &expectedOutput);
    // See if 'requiredInterval' are readily available in the source

    Node::ReadPtr rnode2(node->getChild());
    const Signal::Cache& ss = rnode2->data()->cache;
    ss.samplesDesc ();

    Signal::Intervals i = rnode2->data()->cache.samplesDesc();

    Node::WritePtr wnode(node);
    Node::NodeData* data2 = wnode->data();
    Signal::Operation::Ptr o = data2->operation (e);
    Signal::pBuffer b; // = source data from cache
    Signal::pBuffer out = o->process (b);
/*    {
        DagHead::Ptr p = *i;
        Signal::Intervals I = p->invalid ();
        Node::Ptr node = p->head ();

        getsearchjob(engine, node, I);
    }*/
    return Task();
}


void searchjob(ComputingEngine* engine, Node::Ptr node, Signal::Intervals I)
{
    Signal::Interval required;
    {
        Node::WritePtr r(node);
        Node::NodeData* data = r->data();
        I -= data->current_processing;
        Signal::Operation::Ptr o = data->operation (engine);
        Signal::Interval expectedOutput;
        required = data->operationDesc().requiredInterval (I, &expectedOutput);
    }
/*
    Node::Ptr child = node->getChild();
    {
        Node::ReadPtr c(child);
        Node::NodeData* data = r->data();
        Signal::Intervals missing = required - data->cache.samplesDesc ();
        if (missing.empty ())
        {
            Signahildl::pBuffer b = data->cache.readFixedLength(required);
            {
                Node::R c(child);
                {
        }
    }
            {

            }
        }
        o->process ()
        ->data()-> >operationDesc();
        requiredInterval
        I.fetchInterval ( , c)
    }
    */
    LOG_ERROR("not implemented");
}


/*
void doOneTask()
{
    // Prioritera med round robin.
    for (std::set<DagHead::Ptr>::iterator itr = dag_heads_.begin ();
         itr != dag_heads_.end ();
         ++itr)
    {

    }
    read (const Node &node, Signal::Interval I)
}


Signal::pBuffer Processor::
        read (const Node &node, Signal::Interval I)
{
    Signal::SinkSource& cache = node.data ().cache ();
    Signal::Intervals missing = I - cache.samplesDesc ();

    Signal::Operation::Ptr operation = node.data ().operation (this, computing_engine_);

    while (!missing.empty ())
    {
        Signal::pBuffer r = readSkipCache (node, missing.fetchFirstInterval (), operation);
        if (cache.num_channels () != r->number_of_channels ())
            cache = Signal::SinkSource(r->number_of_channels ());
        cache.put (r);
        missing -= r->getInterval ();

        if (r->getInterval () == I)
            return r;
    }

    return cache.readFixedLength (I);
}


Signal::pBuffer Processor::
        readSkipCache (const Node &node, Signal::Interval I, Signal::Operation::Ptr operation)
{
    const Signal::Interval rI = operation->requiredInterval( I );

    // EXCEPTION_ASSERT(I.count () && node.data().output_buffer->number_of_samples ());

    int N = node.numChildren ();
    Signal::pBuffer b;
    switch(N)
    {
    case 0:
        {
            const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&node.operationDesc ());
            EXCEPTION_ASSERTX( osd, boost::format(
                                   "The first node in the dag was not an instance of "
                                   "OperationSourceDesc but: %1 (%2)")
                               % node.operationDesc ()
                               % vartype(node.operationDesc ()));

            b = Signal::pBuffer( new Signal::Buffer(rI, osd->getSampleRate (), osd->getNumberOfChannels ()));
            b = operation->process (b);
            break;
        }
    case 1:
        b = read (node.getChild (), rI);
        b = operation->process (b);
        break;

    default:
        {
            for (int i=0; i<N; ++i)
            {
                Signal::pBuffer r = read (node.getChild (i), rI);
                if (!b)
                    b = r;
                else
                {
                    Signal::pBuffer n(new Signal::Buffer(b->getInterval (), b->sample_rate (), b->number_of_channels ()+r->number_of_channels ()));
                    unsigned j=0;
                    for (;j<b->number_of_channels ();++j)
                        *n->getChannel (j) |= *b->getChannel (j);
                    for (;j<n->number_of_channels ();++j)
                        *n->getChannel (j) |= *r->getChannel (j-b->number_of_channels ());
                    b = n;
                }
            }

            b = operation->process (b);
            break;
        }
    }

    EXCEPTION_ASSERT_EQUALS( b->getInterval (), I );

    return b;
}
*/

} // namespace Dag
} // namespace Signal
