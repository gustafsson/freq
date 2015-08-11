#ifndef SIGNAL_PROCESSING_WORKER_H
#define SIGNAL_PROCESSING_WORKER_H

#include <memory>
#include <boost/exception_ptr.hpp>
#include "signal/computingengine.h"

namespace Signal {
namespace Processing {

class Worker
{
public:
    typedef std::unique_ptr<Worker> ptr;

    virtual ~Worker() {}

    virtual void abort() = 0;
    virtual bool wait() = 0;
    virtual bool wait(unsigned long time_ms) = 0;
    virtual bool isRunning() = 0;
    virtual std::exception_ptr caught_exception() = 0;
};


class IWorkerFactory {
public:
    typedef std::unique_ptr<IWorkerFactory> ptr;

    virtual ~IWorkerFactory() {}

    virtual Worker::ptr make_worker(Signal::ComputingEngine::ptr ce) = 0;
};

}
}

#endif
