#ifndef SIGNAL_POLLWORKER_WORKER_H
#define SIGNAL_POLLWORKER_WORKER_H

#include "signal/processing/ischedule.h"
#include "signal/computingengine.h"
#include "signal/processing/worker.h"

#include "shared_state.h"
#include "timer.h"
#include "logtickfrequency.h"

#include <QtCore> // QObject, QThread, QPointer

#include <boost/exception/all.hpp>
#include <boost/exception_ptr.hpp>

namespace Signal {
namespace QtEventWorker {

/**
 * @brief The Worker class should run tasks as given by the scheduler.
 *
 * It should wait to be dispatched with a wakeup signal if there are no tasks.
 *
 * It should store information about a crashed task (both segfault and
 * std::exception as well as LockFailed) and stop execution.
 *
 * It should not hang if it causes a deadlock.
 * In the sense that Worker.terminate () still works;
 *
 * It should announce when tasks are finished.
 */
class QtEventWorker: public QObject, public Processing::Worker
{
    Q_OBJECT
public:
    class TerminatedException: virtual public boost::exception, virtual public std::exception {};

    QtEventWorker (Signal::ComputingEngine::ptr computing_eninge, Signal::Processing::ISchedule::ptr schedule, bool wakeuprightaway=true);
    ~QtEventWorker ();

    void abort() override;
    void terminate();
    // wait returns !isRunning
    bool wait() override;
    bool wait(unsigned long time_ms) override;
    bool isRunning() override;
    double activity() override;
    std::exception_ptr caught_exception() override;

signals:
    void oneTaskDone();
    void finished(std::exception_ptr, Signal::ComputingEngine::ptr);

public slots:
    void wakeup();

private slots:
    void finished();

private:
    void loop_while_tasks();

    Signal::ComputingEngine::ptr            computing_engine_;
    Signal::Processing::ISchedule::ptr      schedule_;

    QThread*                                thread_;
    shared_state<std::exception_ptr>        exception_;
    std::exception_ptr                      terminated_exception_;

    LogTickFrequency                        ltf_wakeups_;
    LogTickFrequency                        ltf_tasks_;

    Timer                                   timer_start_;
    double                                  active_time_since_start_;
public:
    static void test ();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_POLLWORKER_WORKER_H
