#include "processor.h"

namespace Signal {
namespace Dag {

Processor::
Processor (QReadWriteLock* lock, Node::Ptr* head_node, ComputingEngine* computing_engine)
    :
      lock_(lock),
      head_node_(head_node),
      computing_engine_(computing_engine)
{
}


Processor::
~Processor()
{
    QWriteLocker l(lock_);
    for (OperationInstances_::iterator itr = operation_instances_.begin ();
         itr != operation_instances_.end ();
         ++itr)
    {
        Node::Ptr n(*itr);
        if (n)
            n->data ().removeOperation (this);
    }
}


Signal::pBuffer Processor::
        read (Signal::Interval I)
{
    QReadLocker l(lock_);
    return read(*head_node_->get (), I);
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
    I = operation->requiredInterval( I );

    // EXCEPTION_ASSERT(I.count () && node.data().output_buffer->number_of_samples ());

    int N = node.numChildren ();
    Signal::pBuffer b;
    switch(N)
    {
    case 0:
        // just a cumbersome way of transferring the informtion in 'I'
        b = Signal::pBuffer( new Signal::Buffer(I, 44100, 1));
        b = operation->process (b);
        break;

    case 1:
        b = read (node.getChild (), I);
        b = operation->process (b);
        break;

    default:
        {
            std::vector<Signal::pBuffer> B( N );
            for (int i=0; i<N; ++i)
                B[i] = read (node.getChild (i), I);

            b = operation->process (B);
            break;
        }
    }

    return b;
}

} // namespace Dag
} // namespace Signal
