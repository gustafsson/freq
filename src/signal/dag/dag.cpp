#include "signaldag.h"
#include "signal/operation.h"
namespace Signal {
namespace Dag {

SignalDag::
        SignalDag (Node::Ptr rootp)
    :
      tip_(rootp),
      root_(rootp)
{
    Node::ReadPtr root(rootp);
    const OperationDesc& desc = root->data()->operationDesc ();
    const Signal::OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&desc);
    EXCEPTION_ASSERTX( osd, boost::format(
                           "The first node in the dag was not an instance of "
                           "OperationSourceDesc but: %1 (%2)")
                       % desc
                       % vartype(desc));
    EXCEPTION_ASSERT_EQUALS( osd->getNumberOfSources (), 0 );
}


int SignalDag::
        sampleRate()
{
    Node::ReadPtr root(root_);
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root->data()->operationDesc ());
    return osd->getSampleRate ();
}


int SignalDag::
        numberOfSamples()
{
    Node::ReadPtr root(root_);
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root->data()->operationDesc ());
    return osd->getNumberOfSamples ();
}


int SignalDag::
        numberOfChannels()
{
    Node::ReadPtr root(root_);
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root->data()->operationDesc ());
    return osd->getNumberOfChannels ();
}

} // namespace Dag
} // namespace Signal
