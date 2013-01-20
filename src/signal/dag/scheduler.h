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

    // Run never returns. But may throw exceptions.
    void run(ComputingEngine*) volatile;

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
    QMutex invalidated_samples_mutex_;
    QWaitCondition invalidated_samples_wait_;

    class Task {
    public:
        Node::Ptr node;
        Signal::pBuffer data;
        Signal::Interval expected_result;

        Task() {}
        Task(Node::Ptr node, Signal::pBuffer data, Signal::Interval expected_result)
            : node(node), data(data), expected_result(expected_result) {}
    };

    Task                runOneIfReady(ComputingEngine*) volatile;
    Task                getNextTask(ComputingEngine*) volatile;
    static Task         searchjob(ComputingEngine* engine, Node::Ptr node, const Signal::Intervals& required);


    std::set<DagHead::Ptr>::iterator dag_head_i_;
    std::set<DagHead::Ptr> dag_heads_;

public:
    static void         test();
};

} // namespace Dag
} // namespace Signal

#endif // SIGNAL_DAG_SCHEDULER_H
