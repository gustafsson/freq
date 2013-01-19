#ifndef SIGNAL_DAG_SCHEDULER_H
#define SIGNAL_DAG_SCHEDULER_H

#include "node.h"
#include "daghead.h"
#include "processor.h"

#include <QThread>
#include <QMutex>
#include <QWaitCondition>

namespace Signal {
namespace Dag {

/**
 * @brief The Dag::Scheduler class schedules execution of processing.
 *
 * Scheduler monitors Dags and Heads and reacts to changes.
 */
class Scheduler: public QObject, public VolatilePtr<Scheduler> {
    Q_OBJECT
public:

    void addDagHead(DagHead::Ptr) volatile;
    void removeDagHead(DagHead::Ptr) volatile;

    void sleepUntilWork() volatile;


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

private slots:
    void invalidatedSamples();

private:
    QMutex task_lock_;
    QWaitCondition task_wait_;

    class Task {
    public:
        Node::Ptr node;
        Signal::Interval I;
    };

    Task        getNextTask(ComputingEngine*) volatile;
    Task        getNextTask(ComputingEngine*);

    void        doOneTask();

    std::set<DagHead::Ptr>::iterator dag_head_i_;
    std::set<DagHead::Ptr> dag_heads_;
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_SCHEDULER_H
