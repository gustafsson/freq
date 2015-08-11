#include "taskworkers.h"
#include "taskworker.h"

#include "thread_pool.h"
#include "signal/computingengine.h"
#include "demangle.h"
#include "expectexception.h"
#include "tasktimer.h"
#include "log.h"

#include <map>
#include <thread>

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

using namespace std;
using namespace JustMisc;
using namespace Signal::Processing;

namespace Signal {
namespace TaskWorker {

TaskWorkers::TaskWorkers(ISchedule::ptr schedule, Bedroom::ptr bedroom)
    : schedule_(schedule),
      bedroom_(bedroom)
{
}


Signal::Processing::Worker::ptr TaskWorkers::
        make_worker(Signal::ComputingEngine::ptr ce)
{
    return TaskWorker::ptr(new TaskWorker(ce, bedroom_, schedule_));
}


} // namespace TaskWorker
} // namespace Signal


namespace Signal {
namespace TaskWorker {

void TaskWorkers::
        test()
{
    Workers::test ([](ISchedule::ptr schedule){
        Bedroom::ptr bedroom(new Bedroom);
        return IWorkerFactory::ptr(new TaskWorkers(schedule, bedroom));
    });
}

} // namespace TaskWorker
} // namespace Signal
