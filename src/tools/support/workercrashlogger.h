#ifndef TOOLS_SUPPORT_WORKERCRASHLOGGER_H
#define TOOLS_SUPPORT_WORKERCRASHLOGGER_H

#include <QObject>
#include <QThread>
#include "signal/processing/workers.h"

namespace Tools {
namespace Support {

/**
 * @brief The WorkerCrashLogger class should fetch information asynchronously
 * of crashed workers.
 */
class WorkerCrashLogger : public QObject
{
    Q_OBJECT
public:
    explicit WorkerCrashLogger(Signal::Processing::Workers::Ptr workers, bool consume_exceptions=true);
    ~WorkerCrashLogger();

private slots:
    void worker_quit(boost::exception_ptr, Signal::ComputingEngine::Ptr);
    void check_all_previously_crashed_and_consume();
    void check_all_previously_crashed_without_consuming();

private:
    Signal::Processing::Workers::Ptr    workers_;
    QThread                             thread_;
    bool                                consume_exceptions_;

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_WORKERCRASHLOGGER_H
