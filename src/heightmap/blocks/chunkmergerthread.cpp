#include "chunkmergerthread.h"

#include "tasktimer.h"
#include "timer.h"
#include "tools/applicationerrorlogcontroller.h"

#include <QGLWidget>

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {

ChunkMergerThread::
        ChunkMergerThread(QGLWidget*shared_gl_context)
    :
      jobs(new Jobs),
      shared_gl_context(shared_gl_context)
{
    // Check for clean exit
    connect(this, SIGNAL(finished()), SLOT(threadFinished()));

    // Start the worker thread as a background thread
    start (LowPriority);
}


ChunkMergerThread::
        ~ChunkMergerThread()
{
    TaskInfo ti("~ChunkMergerThread");

    bool was_idle = isEmpty ();
    requestInterruption ();
    clear ();
    semaphore.release (1);

    if (!was_idle)
      {
        TaskTimer ti("Waiting");
        QThread::wait ();
      }

    QThread::wait ();
}


void ChunkMergerThread::
        clear()
{
    INFO TaskTimer ti("ChunkMergerThread::clear");

    Jobs::WritePtr jobs(this->jobs);

    while (!jobs->empty ())
        jobs->pop ();
}


void ChunkMergerThread::
        addChunk( MergeChunk::Ptr merge_chunk,
                  Tfr::ChunkAndInverse chunk,
                  std::vector<pBlock> intersecting_blocks )
{
    EXCEPTION_ASSERT( merge_chunk );
    EXCEPTION_ASSERT( chunk.chunk );
    EXCEPTION_ASSERT( intersecting_blocks.size () );

    Job j;
    j.merge_chunk = merge_chunk;
    j.chunk = chunk;
    j.intersecting_blocks = intersecting_blocks;

    Jobs::WritePtr jobsw(jobs);
    if (!isInterruptionRequested ())
        jobsw->push (j);

    semaphore.release (1);
}


bool ChunkMergerThread::
        processChunks(float timeout) volatile
{
    WritePtr selfp(this);
    ChunkMergerThread* self = dynamic_cast<ChunkMergerThread*>(&*selfp);

    if (0 <= timeout)
      {
        // return immediately
        return self->isEmpty ();
      }

    // Requested wait until done
    return self->wait(timeout);
}


bool ChunkMergerThread::
        wait(float timeout)
{
    Timer T;

    if (timeout < 0)
        timeout = FLT_MAX;

    bool empty;
    while (!(empty = isEmpty ()) && T.elapsed () < timeout && isRunning ())
      {
        QThread::wait (5); // Sleep 5 ms
      }

    return empty;
}


void ChunkMergerThread::
        threadFinished()
{
    TaskInfo("ChunkMergerThread::threadFinished");

    try {
        EXCEPTION_ASSERTX(isInterruptionRequested (), "Thread quit unexpectedly");
        EXCEPTION_ASSERTX(isEmpty(), "Thread quit with jobs left");
    } catch (...) {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
    }
}


bool ChunkMergerThread::
        isEmpty() const
{
    return read1(jobs)->empty();
}


void ChunkMergerThread::
        run()
{
    try
      {
        QGLWidget w(0, shared_gl_context);
        w.makeCurrent ();

        while (!isInterruptionRequested ())
          {
            while (semaphore.tryAcquire ()) {}

            while (true)
              {
                Job job;
                  {
                    Jobs::WritePtr jobsr(jobs);
                    if (jobsr->empty ())
                        break;
                    job = jobsr->front ();
                  }

                processJob (job);

                  {
                    // Want processChunks(-1) and self->isEmpty () to return false until
                    // the job has finished processing.

                    Jobs::WritePtr jobsw(jobs);
                    // Both 'clear' and 'addChunk' may have been called in between, so only
                    // pop the queue if the first job is still the same.
                    if (!jobsw->empty() && job.chunk.chunk == jobsw->front().chunk.chunk)
                        jobsw->pop ();

                    // Release OpenGL resources before releasing the memory held by chunk
                    job.merge_chunk.reset ();
                  }
              }

            // Make sure any texture upload is complete
            {
                INFO TaskTimer tt("glFinish");
                glFinish ();
            }

            semaphore.acquire ();
          }
      }
    catch (...)
      {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
      }
}


void ChunkMergerThread::
        processJob(Job& j)
{
    std::vector<IChunkToBlock::Ptr> chunk_to_blocks = write1( j.merge_chunk )->createChunkToBlock( j.chunk );

    for (IChunkToBlock::Ptr chunk_to_block : chunk_to_blocks)
      {
        for (pBlock block : j.intersecting_blocks)
          {
            if (!block->frame_number_last_used)
                continue;

            INFO TaskTimer tt(boost::format("block %s") % block->getRegion ());
            chunk_to_block->mergeChunk (block);
          }
      }
}


} // namespace Blocks
} // namespace Heightmap
