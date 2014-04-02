#ifndef TOOLS_SUPPORT_WORKERCRASHLOGGER_H
#define TOOLS_SUPPORT_WORKERCRASHLOGGER_H

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
    explicit WorkerCrashLogger(Signal::Processing::Workers::ptr workers, bool consume_exceptions=true);
    ~WorkerCrashLogger();

private slots:
    void worker_quit(std::exception_ptr, Signal::ComputingEngine::ptr);
    void check_all_previously_crashed_and_consume();
    void check_all_previously_crashed_without_consuming();
    void finished();

private:
    Signal::Processing::Workers::ptr    workers_;
    QThread                             thread_;
    bool                                consume_exceptions_;

    void log(const boost::exception& x);

public:
    static void test();
};

} // namespace Support
} // namespace Tools

#endif // TOOLS_SUPPORT_WORKERCRASHLOGGER_H
