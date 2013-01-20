#include "processor.h"

namespace Signal {
namespace Dag {

Processor::
Processor (Node::Ptr* head_node, ComputingEngine* computing_engine)
    :
      head_node_(head_node),
      computing_engine_(computing_engine)
{
}


Processor::
        ~Processor()
{
    for (OperationInstances_::iterator itr = operation_instances_.begin ();
         itr != operation_instances_.end ();
         ++itr)
    {
        Node::Ptr n(*itr);
        if (n)
            LOG_ERROR("Not implemented");
//            Node::WritePtr (n)->data ()->removeOperation (this);
    }
}


Signal::pBuffer Processor::
        read (Signal::Interval I)
{
    return read(*head_node_, I);
}


Signal::pBuffer Processor::
        read (Node::Ptr nodep, Signal::Interval I)
{
    Node::WritePtr nodew(nodep);
    Node::NodeData* data = nodew->data();
    Signal::Intervals missing = I - data->cache.samplesDesc ();

    Signal::Operation::Ptr operation = data->operation ( computing_engine_);

    while (!missing.empty ())
    {
        LOG_ERROR("not implemented");
        Signal::pBuffer r = readSkipCache (*nodew, missing.fetchFirstInterval (), operation);
        if (data->cache.num_channels () != (int)r->number_of_channels ())
            data->cache = Signal::Cache();//r->number_of_channels ());
        data->cache.put (r);
        missing -= r->getInterval ();

        if (r->getInterval () == I)
            return r;
    }

    return data->cache.read (I);
}


Signal::pBuffer Processor::
        readSkipCache (Node &node, Signal::Interval I, Signal::Operation::Ptr operation)
{
    Signal::Interval expectedOutput;
    LOG_ERROR("Not implemented");
    const Signal::Interval rI;// = operation->requiredInterval( I, &expectedOutput );

    // EXCEPTION_ASSERT(I.count () && node.data().output_buffer->number_of_samples ());

    int N = node.numChildren ();
    Signal::pBuffer b;
    switch(N)
    {
    case 0:
        {
            const Signal::OperationDesc& desc = node.data ()->operationDesc ();
            const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&desc);
            EXCEPTION_ASSERTX( osd, boost::format(
                                   "The first node in the dag was not an instance of "
                                   "OperationSourceDesc but: %1 (%2)")
                               % desc
                               % vartype(desc));

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

    EXCEPTION_ASSERT_EQUALS( b->getInterval (), expectedOutput );

    return b;
}


} // namespace Dag
} // namespace Signal
