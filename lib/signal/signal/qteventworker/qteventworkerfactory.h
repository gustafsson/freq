#ifndef SIGNAL_POLLWORKER_WORKERS_H
#define SIGNAL_POLLWORKER_WORKERS_H

#include "qteventworker.h"
#include "bedroomsignaladapter.h"
#include "signal/processing/ischedule.h"
#include "signal/computingengine.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/workers.h"

#include <vector>
#include <map>
#include <QObject>

namespace Signal {
namespace QtEventWorker {

/**
 * @brief The Schedule class should start and stop computing engines as they
 * are added and removed.
 *
 * A started engine is a thread that uses class Worker which queries a GetTask
 * for tasks to work on.
 *
 * It should terminate all threads when it's closed.
 *
 * It should wake up sleeping workers when any work is done to see if they can
 * help out on what's left.
 */
class QtEventWorkerFactory: public QObject, public Signal::Processing::IWorkerFactory
{
    Q_OBJECT
public:
    QtEventWorkerFactory(Signal::Processing::ISchedule::ptr schedule, Signal::Processing::Bedroom::ptr bedroom);
    ~QtEventWorkerFactory();

    Signal::Processing::Worker::ptr make_worker(Signal::ComputingEngine::ptr ce) override;

    /**
     * @brief terminate_workers terminates all worker threads and doesn't
     * return until all threads have been terminated.
     *
     * Use of this function is discouraged because it doesn't allow threads
     * to clean up any resources nor to release any locks.
     *
     * The thread might not terminate until it activates the OS scheduler by
     * entering a lock or going to sleep.
     *
     * Returns true if all threads were terminated within 'timeout'.
     */
    static bool terminate_workers(Processing::Workers& workers, int timeout=1000);

signals:
    void worker_quit(std::exception_ptr, Signal::ComputingEngine::ptr);

private:
    class WorkerWrapper : public Signal::Processing::Worker {
    public:
        WorkerWrapper(QtEventWorker* p);
        ~WorkerWrapper();

        void abort() override {p->abort();}
        bool wait() override {return p->wait();}
        bool wait(unsigned long time_ms) override {return p->wait(time_ms);}
        bool isRunning() override {return p->isRunning();}
        double activity() override {return p->activity();}
        std::exception_ptr caught_exception() override {return p->caught_exception();}

        void terminate() {p->terminate ();}

    private:
        QtEventWorker* p; // Managed by Qt
    };

    Processing::ISchedule::ptr schedule_;
    Processing::Bedroom::ptr bedroom_;
    BedroomSignalAdapter* notifier_;

public:
    static void test();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_POLLWORKER_WORKERS_H
