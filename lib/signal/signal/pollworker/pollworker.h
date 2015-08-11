#ifndef SIGNAL_POLLWORKER_WORKER_H
#define SIGNAL_POLLWORKER_WORKER_H

#include "signal/processing/ischedule.h"
#include "signal/computingengine.h"
#include "signal/processing/worker.h"

#include "shared_state.h"
#include "timer.h"

#include <QtCore> // QObject, QThread, QPointer

#include <boost/exception/all.hpp>
#include <boost/exception_ptr.hpp>

namespace Signal {
namespace PollWorker {

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
class PollWorker: public QObject, public Processing::Worker
{
    Q_OBJECT
public:
    class TerminatedException: virtual public boost::exception, virtual public std::exception {};

    PollWorker (Signal::ComputingEngine::ptr computing_eninge, Signal::Processing::ISchedule::ptr schedule, bool wakeuprightaway=true);
    ~PollWorker ();

    void abort() override;
    void terminate();
    // wait returns !isRunning
    bool wait() override;
    bool wait(unsigned long time_ms) override;
    bool isRunning() override;

    // 'if (caught_exception())' will be true if an exception was caught.
    //
    // To examine the exception. Use this pattern:
    //
    //     try {
    //         rethrow_exception(caught_exception ());
    //     } catch ( std::exception& x ) {
    //         x.what();
    //         ... get_error_info<...>(x);
    //         boost::diagnostic_information(x);
    //     }
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

    Timer                                   timer_;
    int                                     wakeups_ = 0;
    double                                  next_tick_ = 10;
public:
    static void test ();
};


} // namespace Processing
} // namespace Signal

#endif // SIGNAL_POLLWORKER_WORKER_H
