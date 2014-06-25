#include "updateconsumer.h"
#include "updatequeue.h"

#include "waveformblockupdater.h"
#include "tfrblockupdater.h"
#include "heightmap/uncaughtexception.h"
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
namespace Update {

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
        Heightmap::UncaughtException::handle_exception(boost::current_exception());
    }
}


template <class Q>
    typename Q::container_type& Container(Q& q) {
        struct HackedQueue : private Q {
            static typename Q::container_type& Container(Q& q) {
                return q.*&HackedQueue::c;
            }
        };
    return HackedQueue::Container(q);
}


void UpdateConsumer::
        run()
{
    QGLWidget w(0, shared_gl_context);
    w.makeCurrent ();

    try
      {
        TfrBlockUpdater block_updater;
        WaveformBlockUpdater waveform_updater;

        while (!isInterruptionRequested ())
          {
            unique_ptr<TaskTimer> tt;
            INFO if (update_queue->empty ())
                tt.reset (new TaskTimer("Waiting for updates"));
            UpdateQueue::Job j = update_queue->pop ();
            tt.reset ();

            auto jobqueue = update_queue->clear ();

            // Force a push_front to the std::queue
            Container(jobqueue).push_front (std::move(j));

//            Timer t;

            while (!jobqueue.empty ())
            {
                unsigned s = jobqueue.size ();
                block_updater.processJobs (jobqueue);
                waveform_updater.processJobs (jobqueue);
                EXCEPTION_ASSERT_LESS(jobqueue.size (), s);
            }

            if (!isInterruptionRequested ())
              {
                emit didUpdate ();
              }

/*            INFO
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
            */
        }
      }
    catch (UpdateQueue::abort_exception&)
      {
      }
    catch (...)
      {
        Heightmap::UncaughtException::handle_exception(boost::current_exception());
      }
}


} // namespace Update
} // namespace Heightmap
