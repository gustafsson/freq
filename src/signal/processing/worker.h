#ifndef SIGNAL_PROCESSING_WORKER_H
#define SIGNAL_PROCESSING_WORKER_H

#include "ischedule.h"
#include "signal/computingengine.h"

#include <QThread>
#include <QPointer>

#include <boost/exception_ptr.hpp>

namespace Signal {
namespace Processing {

/**
 * @brief The Worker class should run the next task as long as there is one
 *
 * It should store information about a crashed task (both segfault and std::exception)
 *
 * Issues
 * should not subclass QThread
 * http://mayaposch.wordpress.com/2011/11/01/how-to-really-truly-use-qthreads-the-full-explanation/
 */
class Worker
        : public QThread
{
public:
    // This is a Qt object that can delete itself, and as such we shall not use
    // boost::shared_ptr but the Qt smart pointer QPointer which is aware of Qt
    // objects that delete themselves.
    typedef QPointer<Worker> Ptr;

    Worker (Signal::ComputingEngine::Ptr computing_eninge, ISchedule::WeakPtr schedule);

    // Delete when finished
    virtual void run ();

    // Postpones the thread exit until a task has been finished.
    // 'scheduel_->getTask()' might be idling but this class is unaware of that
    void exit_nicely_and_delete();

    // 'if (caught_exception())' will be true if an exception was caught.
    //
    // To examine the exception. Use this pattern:
    //
    //     try {
    //         rethrow_exception(caught_exception ());
    //     } catch ( std::exception& x ) {
    //         x.what();
    //         ... get_error_info<...>(x);
    //     }
    boost::exception_ptr caught_exception() const;

private:
    Signal::ComputingEngine::Ptr computing_eninge_;
    ISchedule::WeakPtr schedule_;

    boost::exception_ptr exception_;

public:
    static void test ();
};

} // namespace Processing
} // namespace Signal

#endif // SIGNAL_PROCESSING_WORKER_H
