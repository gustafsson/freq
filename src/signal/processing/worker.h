#ifndef SIGNAL_PROCESSING_WORKER_H
#define SIGNAL_PROCESSING_WORKER_H

#include <QThread>
#include <QWaitCondition>
#include "signal/computingengine.h"

namespace Signal {
namespace Processing {

class Worker
        : public QThread
{
public:
    Worker (Signal::ComputingEngine::Ptr computing_eninge);

    virtual void run ();
/*
 *move to schedule
    QMutex _todo_lock;
    QMutex _work_lock;
    QWaitCondition work_condition;
*/
private:
    Signal::ComputingEngine::Ptr computing_eninge_;

public:
    static void test ();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKER_H
