#include "dag.h"
#include "signal/operation.h"
namespace Signal {
namespace Dag {

Dag::
        Dag (Node::Ptr root)
    :
      tip_(root),
      root_(root)
{
    const Signal::OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root->operationDesc ());
    EXCEPTION_ASSERTX( osd, boost::format(
                           "The first node in the dag was not an instance of "
                           "OperationSourceDesc but: %1 (%2)")
                       % root->operationDesc ()
                       % vartype(root->operationDesc ()));
    EXCEPTION_ASSERT_EQUALS( osd->getNumberOfSources (), 0 );
}


int Dag::
        sampleRate()
{
    QReadLocker l (root_->lock ());
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root_->operationDesc ());
    return osd->getSampleRate ();
}


int Dag::
        numberOfSamples()
{
    QReadLocker l (root_->lock ());
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root_->operationDesc ());
    return osd->getNumberOfSamples ();
}


int Dag::
        numberOfChannels()
{
    QReadLocker l (root_->lock ());
    const OperationSourceDesc* osd = dynamic_cast<const OperationSourceDesc*>(&root_->operationDesc ());
    return osd->getNumberOfChannels ();
}

} // namespace Dag
} // namespace Signal
