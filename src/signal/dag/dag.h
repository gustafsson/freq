#ifndef SIGNAL_DAG_H
#define SIGNAL_DAG_H

// DAG, Directed_acyclic_graph
// http://en.wikipedia.org/wiki/Directed_acyclic_graph
//
// Describes a graph of signal processing nodes and operations on it.

#include "dagcommand.h"
#include "processor.h"

#include <QReadWriteLock>

namespace Signal {
namespace Dag {

class Dag
{
public:
    typedef boost::shared_ptr<Dag> Ptr;

    Dag (Node::Ptr head);

    Processor getProcessor();
    void queueCommand(ICommand::Ptr cmd);
private:
    void executeQueue();

    QReadWriteLock dag_lock_;
    Node::Ptr head_; // There's only one 'head'.
    Node::Ptr tip_; // There's only one 'tip'. 'head' and 'tip' might be the same.
    // Multiple Dags are out-of-scope.
    // Multiple tips are out-of-scope. Or?

    QReadWriteLock cmdqueue_lock_;
    std::vector<ICommand::Ptr> cmdqueue_;
};


} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_H
