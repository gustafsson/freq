#include "updateconsumer.h"
#include "updatequeue.h"

#include "tools/applicationerrorlogcontroller.h"
#include "heightmap/tfrmappings/waveformblockfilter.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"

#include <QGLWidget>

#include <numeric>

//#define INFO
#define INFO if(0)

using namespace std;

namespace Heightmap {
namespace Blocks {

UpdateConsumer::
        UpdateConsumer(QGLWidget*shared_gl_context, UpdateQueue::ptr update_queue)
    :
      QThread(shared_gl_context),
      shared_gl_context(shared_gl_context),
      update_queue(update_queue)
{
    // Check for clean exit
    connect(this, SIGNAL(finished()), SLOT(threadFinished()));

    // Start the worker thread as a background thread
    start (LowPriority);
}


UpdateConsumer::
        ~UpdateConsumer()
{
    TaskInfo ti ("~UpdateConsumer");

    requestInterruption ();
    update_queue->abort_on_empty ();
    update_queue->clear ();

    if (QThread::isRunning ())
      {
        TaskTimer ti("Waiting");
        QThread::wait ();
      }
    else
        QThread::wait ();
}


void UpdateConsumer::
        threadFinished()
{
    TaskInfo("UpdateConsumer::threadFinished");

    try {
        EXCEPTION_ASSERTX(isInterruptionRequested (), "Thread quit unexpectedly");
        EXCEPTION_ASSERTX(update_queue->empty(), "Thread quit with jobs left");
    } catch (...) {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
    }
}


void UpdateConsumer::
        run()
{
    QGLWidget w(0, shared_gl_context);
    w.makeCurrent ();

    try
      {
        BlockUpdater block_updater;

        while (!isInterruptionRequested ())
          {
            std::unique_ptr<TaskTimer> tt;
            INFO if (update_queue->empty ())
                tt.reset (new TaskTimer("Waiting for updates"));
            UpdateQueue::Job j = update_queue->pop ();
            tt.reset ();
            queue<UpdateQueue::Job> jobqueue = update_queue->clear ();

            Timer t;

            vector<UpdateQueue::Job> jobs;
            jobs.reserve (1 + jobqueue.size ());

            jobs.push_back (move(j));
            while (!jobqueue.empty ())
              {
                jobs.push_back (move(jobqueue.front ()));
                jobqueue.pop ();
              }

            block_updater.processJobs (jobs);
            TfrMappings::WaveformBlockUpdater().processJobs (jobs);

            if (!isInterruptionRequested ())
              {
                emit didUpdate ();
              }

            for (UpdateQueue::Job& j : jobs)
                j.promise.set_value ();

            INFO
            {
                Signal::Intervals span = accumulate(jobs.begin (), jobs.end (), Signal::Intervals(),
                        [](Signal::Intervals& I, const UpdateQueue::Job& j) {
                            if (!j.updatejob)
                                return I;
                            return I|=j.updatejob->getCoveredInterval();
                        });

                std::set<pBlock> blocks;
                for (auto& j : jobs)
                    if (j.updatejob)
                        for (pBlock b : j.intersecting_blocks)
                            blocks.insert(b);

                Log("Updated %d chunks -> %d blocks. %s. %.0f samples/s")
                         % jobs.size () % blocks.size ()
                         % span % (span.count ()/t.elapsed ());
            }
        }
      }
    catch (UpdateQueue::abort_exception&)
      {
      }
    catch (...)
      {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
      }
}


} // namespace Blocks
} // namespace Heightmap
