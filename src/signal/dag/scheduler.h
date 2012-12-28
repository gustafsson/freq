#ifndef SIGNAL_DAG_SCHEDULER_H
#define SIGNAL_DAG_SCHEDULER_H

#include "node.h"
#include "daghead.h"
#include "processor.h"

#include <QThread>

namespace Signal {
namespace Dag {

/**
 * @brief The Dag::Scheduler class schedules execution of processing.
 *
 * Scheduler monitors Dags and Heads and reacts to changes.
 */
class Scheduler: public QThread {
public:
    typedef boost::shared_ptr<Scheduler> Ptr;

    void addComputingEngine(ComputingEngine*);
    void removeComputingEngine(ComputingEngine*);

    void addDagHead(DagHead::Ptr);
    void removeDagHead(DagHead::Ptr);


/*
+++++++------------++++++++++
       ||||||||||||
+++++++++++++++++++++++++++++


+++++++------------++++++++++
       ||||||||||||
       ++++++++++++
+++++++------------++++++++++


+++++++xxxxxxxxxxxx++++++++++
+++++++------------++++++++++


-----------------------------
|||||||||
       ||||||||||||
                 ||||||||||||
+++++++++++++++++++++++++++++

*/


private:
    class Task {
    public:
        Node::Ptr node;
        Signal::Interval I;
    };

    void        run();

    void        doOneTask();
    Task        getNextTask();

    QReadWriteLock lock_;
    std::map<ComputingEngine*, Processor::Ptr> processors_;
    std::set<DagHead::Ptr> dag_heads_;
    std::set<Task> running_tasks_;
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_SCHEDULER_H
