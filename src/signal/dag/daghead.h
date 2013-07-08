#ifndef SIGNAL_DAG_DAGHEAD_H
#define SIGNAL_DAG_DAGHEAD_H

#include "signaldag.h"
#include "signal/operation.h"

#include <QObject>

namespace Signal {
namespace Dag {

class DagHead: public QObject, public VolatilePtr<DagHead>
{
    Q_OBJECT
public:

    DagHead(SignalDag::Ptr dag, Signal::OperationDesc::Ptr headprocessor);
    DagHead(DagHead::Ptr daghead, Signal::OperationDesc::Ptr headprocessor);

    // Used by processor
    Node::Ptr head() const { return head_; }
    SignalDag::Ptr dag() const { return dag_; }

    void queueCommand(ICommand::Ptr cmd);
    void executeQueue();

    void setInvalidSamples(Signal::Intervals invalid);

    Signal::Intervals invalidSamples() const { return invalid_samples_; }

signals:
    void invalidatedSamples();

private:

    QWaitCondition head_modified_;

    Signal::OperationDesc::Ptr head_processor_;
    Signal::Intervals invalid_samples_;

    SignalDag::Ptr dag_;

    // This Node::Ptr is a copy of the actual node in the dag.
    Node::Ptr head_;

    QReadWriteLock cmdqueue_lock_;
    std::vector<ICommand::Ptr> cmdqueue_;
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_DAGHEAD_H
