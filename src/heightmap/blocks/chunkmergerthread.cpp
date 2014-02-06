#include "chunkmergerthread.h"

#include "TaskTimer.h"
#include "timer.h"

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
    // Start the worker thread as a background thread
    start (LowPriority);
}


ChunkMergerThread::
        ~ChunkMergerThread()
{
    clear ();
    requestInterruption ();
    semaphore.release (1);
    QThread::wait ();
}


void ChunkMergerThread::
        clear()
{
    Jobs::WritePtr jobsp(this->jobs);
    std::queue<Job>& jobs = jobsp->jobs;

    while (!jobs.empty ())
        jobs.pop ();
}


void ChunkMergerThread::
        addChunk( MergeChunk::Ptr merge_chunk,
                  Tfr::ChunkAndInverse chunk,
                  std::vector<pBlock> intersecting_blocks )
{
    Job j;
    j.merge_chunk = merge_chunk;
    j.chunk = chunk;
    j.intersecting_blocks = intersecting_blocks;
    write1(jobs)->jobs.push (j);
    semaphore.release (1);
}


bool ChunkMergerThread::
        processChunks(float timeout) volatile
{
    WritePtr selfp(this);
    ChunkMergerThread* self = dynamic_cast<ChunkMergerThread*>(&*selfp);

    if (0 <= timeout) {
        // return immediately
        return self->isEmpty ();
    }

    // Requested wait until done
    return self->wait(timeout);
}


bool ChunkMergerThread::
        wait(float timeout) const
{
    Timer T;
    bool has_data;

    if (timeout < 0)
        timeout = FLT_MAX;

    for (has_data = isEmpty ();
         has_data && T.elapsed () < timeout && isRunning ();
         has_data = isEmpty ())
    {
        QSemaphore().tryAcquire (1, 5); // Sleep 5 ms
    }

    return !has_data;
}


bool ChunkMergerThread::
        isEmpty() const
{
    return read1(jobs)->jobs.empty();
}


void ChunkMergerThread::
        run()
{
    QGLWidget w(0, shared_gl_context);
    w.makeCurrent ();

    while (!isInterruptionRequested ())
      {
        while (!isEmpty ())
          {
            Job job = read1(jobs)->jobs.front ();

            processJob (job);

            write1(jobs)->jobs.pop ();
          }

        semaphore.acquire ();
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
            INFO TaskTimer tt(boost::format("block %s") % block->getRegion ());
            chunk_to_block->mergeChunk (block);
          }
      }
}


} // namespace Blocks
} // namespace Heightmap
