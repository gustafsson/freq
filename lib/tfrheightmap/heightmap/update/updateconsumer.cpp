#include "updateconsumer.h"
#include "updatequeue.h"

#include "waveformblockupdater.h"
#include "tfrblockupdater.h"
#include "heightmap/uncaughtexception.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"
#include "gl.h"

#include <QGLWidget>
#include <QGLContext>
#include <QOpenGLContext>
#include <QOffscreenSurface>

#include <numeric>

//#define INFO
#define INFO if(0)

using namespace std;

namespace Heightmap {
namespace Update {

UpdateConsumer::
        UpdateConsumer(QGLWidget* shared_opengl_widget,
                       UpdateQueue::ptr update_queue)
    :
      UpdateConsumer( shared_opengl_widget->context ()->contextHandle (),
                      update_queue,
                      shared_opengl_widget)
{
}


UpdateConsumer::
        UpdateConsumer(QOpenGLContext* shared_opengl_context,
                       UpdateQueue::ptr update_queue,
                       QObject* parent)
    :
      QThread(parent),
      shared_opengl_context(shared_opengl_context),
      update_queue(update_queue)
{
    EXCEPTION_ASSERT(shared_opengl_context);

    // Check for clean exit
    connect(this, SIGNAL(finished()), SLOT(threadFinished()));

    surface = new QOffscreenSurface;
    surface->setParent (this);
    surface->setFormat (shared_opengl_context->format ());
    surface->create ();

    // Start this worker thread as a background thread
    start (LowPriority);
}



UpdateConsumer::
        ~UpdateConsumer()
{
    requestInterruption ();
    update_queue->abort_on_empty ();
    update_queue->clear ();
    QThread::wait ();
}


void UpdateConsumer::
        threadFinished()
{
    TaskInfo ti("UpdateConsumer::threadFinished");

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
    try
      {
        QOpenGLContext context;
        context.setShareContext (shared_opengl_context);
        context.setFormat (shared_opengl_context->format ());
        context.create ();

        if (!context.shareContext ()) {
            Log("!!! Couldn't share contexts. UpdateConsumer thread is stopped.");
            return;
        }

        context.makeCurrent (surface);

        work ();
      }
    catch (UpdateQueue::abort_exception&)
      {
        requestInterruption ();
      }
    catch (...)
      {
        Heightmap::UncaughtException::handle_exception(boost::current_exception());
        requestInterruption ();
      }
}


void UpdateConsumer::
        work()
{
    TfrBlockUpdater block_updater;
    WaveformBlockUpdater waveform_updater;

    while (!isInterruptionRequested ())
      {
        QCoreApplication::processEvents();

        try
          {
            unique_ptr<TaskTimer> tt;
            INFO if (update_queue->empty ())
                tt.reset (new TaskTimer("Waiting for updates"));
            UpdateQueue::Job j = update_queue->pop ();
            tt.reset ();

            auto jobqueue = update_queue->clear ();

            // Force a push_front to the std::queue
            Container(jobqueue).push_front (std::move(j));

            unsigned num_jobs = jobqueue.size ();
            Timer t;

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


            INFO Log("UpdateConsumer did %d jobs in %s")
                     % num_jobs % TaskTimer::timeToString (t.elapsed ());

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
        catch (UpdateQueue::skip_job_exception&)
          {
          }
      }
}


} // namespace Update
} // namespace Heightmap
