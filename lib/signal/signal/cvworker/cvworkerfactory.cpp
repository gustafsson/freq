#include "cvworkerfactory.h"
#include "cvworker.h"

#include "signal/processing/workers.h"

#include <map>
#include <thread>

//#define TIME_TERMINATE
#define TIME_TERMINATE if(0)

//#define UNITTEST_STEPS
#define UNITTEST_STEPS if(0)

using namespace std;
using namespace Signal::Processing;

namespace Signal {
namespace CvWorker {

CvWorkerFactory::CvWorkerFactory(ISchedule::ptr schedule, Bedroom::ptr bedroom)
    : schedule_(schedule),
      bedroom_(bedroom)
{
}


Signal::Processing::Worker::ptr CvWorkerFactory::
        make_worker(Signal::ComputingEngine::ptr ce)
{
    return CvWorker::ptr(new CvWorker(ce, bedroom_, schedule_));
}

} // namespace CvWorker
} // namespace Signal


namespace Signal {
namespace CvWorker {

void CvWorkerFactory::
        test()
{
    Workers::test ([](ISchedule::ptr schedule){
        Bedroom::ptr bedroom(new Bedroom);
        return IWorkerFactory::ptr(new CvWorkerFactory(schedule, bedroom));
    });
}

} // namespace CvWorker
} // namespace Signal
