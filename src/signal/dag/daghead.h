#ifndef SIGNAL_DAG_DAGHEAD_H
#define SIGNAL_DAG_DAGHEAD_H

#include "dag.h"
#include "signal/operation.h"

namespace Signal {
namespace Dag {

class DagHead
{
public:
    typedef boost::shared_ptr<DagHead> Ptr;

    DagHead(Dag::Ptr dag, Signal::OperationDesc::Ptr headprocessor);
    DagHead(DagHead::Ptr daghead, Signal::OperationDesc::Ptr headprocessor);

    // Used by processor
    Node::Ptr head() { return head_; }
    Dag::Ptr dag() { return dag_; }

    void queueCommand(ICommand::Ptr cmd);

    void setInvalidSamples(Signal::Intervals invalid, Signal::IntervalType center=0);

private:
    void executeQueue();

    QWaitCondition head_modified_;

    Signal::OperationDesc::Ptr head_processor_;

    Dag::Ptr dag_;

    // This Node::Ptr is a copy of the actual node in the dag.
    Node::Ptr head_;

    QReadWriteLock cmdqueue_lock_;
    std::vector<ICommand::Ptr> cmdqueue_;
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_DAGHEAD_H
