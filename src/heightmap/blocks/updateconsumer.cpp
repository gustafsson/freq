#include "updateconsumer.h"
#include "updatequeue.h"

#include "tasktimer.h"
#include "timer.h"
#include "tools/applicationerrorlogcontroller.h"
#include "tfr/chunk.h"

#include <QGLWidget>

//#define INFO
#define INFO if(0)

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
        EXCEPTION_ASSERTX(update_queue->isEmpty(), "Thread quit with jobs left");
    } catch (...) {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
    }
}


void UpdateConsumer::
        run()
{
    try
      {
        QGLWidget w(0, shared_gl_context);
        w.makeCurrent ();

        while (!isInterruptionRequested ())
          {
            UpdateQueue::Job job = update_queue->getJob ();

            if (job.merge_chunk)
            {
                block_updater.processJob (job.merge_chunk, job.chunk, job.intersecting_blocks);

                // Release OpenGL resources before releasing the memory held by chunk
                job.merge_chunk = MergeChunk::ptr ();
            }
          }
      }
    catch (...)
      {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
      }
}


} // namespace Blocks
} // namespace Heightmap
