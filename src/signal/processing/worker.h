#ifndef SIGNAL_PROCESSING_WORKER_H
#define SIGNAL_PROCESSING_WORKER_H

#include "gettask.h"
#include "signal/computingengine.h"

#include <QThread>
#include <QPointer>

namespace Signal {
namespace Processing {

class Worker
        : public QThread
{
public:
    // This is a Qt object that can delete itself, and as such we shall not use
    // boost::shared_ptr but the Qt smart pointer QPointer which is aware of Qt
    // objects that delete themselves.
    typedef QPointer<Worker> Ptr;

    Worker (Signal::ComputingEngine::Ptr computing_eninge, GetTask::Ptr get_task);

    // Delete when finished
    virtual void run ();

    // Postpones the thread exit until a task has been finished.
    // 'get_task_->getTask()' might be idling but this class is unaware of that
    void exit_nicely_and_delete();

private:
    Signal::ComputingEngine::Ptr computing_eninge_;
    GetTask::Ptr get_task_;
    bool enough_;

public:
    static void test ();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKER_H
