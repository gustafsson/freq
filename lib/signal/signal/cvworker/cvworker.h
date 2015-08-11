#ifndef TASKWORKER_H
#define TASKWORKER_H

#include "signal/computingengine.h"
#include "signal/processing/ischedule.h"
#include "signal/processing/bedroom.h"
#include "signal/processing/worker.h"
#include <boost/exception/exception.hpp>

#include <thread>
#include <future>
#include <atomic>

namespace Signal {
namespace CvWorker {

/**
 * @brief The CvWorker class should execute a task as specified by a scheduler
 * and wait on a condition variable (Bedroom) when idle.
 */
class CvWorker: public Processing::Worker
{
public:
    CvWorker(
            Signal::ComputingEngine::ptr computing_eninge,
            Signal::Processing::Bedroom::ptr bedroom,
            Signal::Processing::ISchedule::ptr schedule);
    ~CvWorker();

    void abort() override;
    // wait returns !isRunning
    bool wait() override;
    bool wait(unsigned long ms) override;
    bool isRunning() override;
    std::exception_ptr caught_exception() override;

private:
    void join();
    std::thread t;
    std::future<void> f;

    std::atomic<bool>                   abort_;
    Signal::Processing::Bedroom::ptr    bedroom_;
    std::exception_ptr                  caught_exception_;

public:
    static void test ();
};

} // namespace CvWorker
} // namespace Signal

#endif // TASKWORKER_H
