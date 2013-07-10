#include "worker.h"

namespace Signal {
namespace Processing {

Worker::
        Worker (Signal::ComputingEngine::Ptr computing_eninge)
    :
      computing_eninge_(computing_eninge)
{

}


void Worker::
        run()
{

}


void Worker::
        test()
{
    // It should run the next task as long as there is a task
    {
        Signal::ComputingEngine::Ptr computing_eninge(new Signal::ComputingCpu);
        Worker worker(computing_eninge);
        worker.run ();
    }
}


} // namespace Processing
} // namespace Signal
