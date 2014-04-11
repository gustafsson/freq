#include "updateconsumer.h"
#include "updatequeue.h"

#include "tasktimer.h"
#include "timer.h"
#include "tools/applicationerrorlogcontroller.h"
#include "heightmap/tfrmappings/waveformblockfilter.h"
#include "tfr/chunk.h"

#include <numeric>

#include <QGLWidget>

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

        typedef shared_ptr<ChunkToBlockDegenerateTexture::DrawableChunk> pDrawableChunk;

//        typedef pair<pDrawableChunk,vector<pBlock>> chunk_with_blocks_t;
//        vector<chunk_with_blocks_t> chunks_with_blocks;

        ChunkToBlockDegenerateTexture::BlockFbos& block_fbos = block_updater.block_fbos ();

        while (!isInterruptionRequested ())
          {
            std::unique_ptr<TaskTimer> tt;
//            if (update_queue->empty ())
//                tt.reset (new TaskTimer("Waiting for updates"));
            UpdateQueue::Job j = update_queue->pop ();
            tt.reset ();
            queue<UpdateQueue::Job> jobqueue = update_queue->clear ();
            map<pBlock, queue<pDrawableChunk>> chunks_per_block;

            Timer t;

            vector<UpdateQueue::Job> jobs;
            jobs.reserve (1 + jobqueue.size ());

            jobs.push_back (move(j));
            while (!jobqueue.empty ())
              {
                jobs.push_back (move(jobqueue.front ()));
                jobqueue.pop ();
              }

            Signal::Intervals span = accumulate(jobs.begin (), jobs.end (), Signal::Intervals(),
                    [](Signal::Intervals& I, const UpdateQueue::Job& j) {
                        if (!j.updatejob)
                            return I;
                        return I|=j.updatejob->getCoveredInterval();
                    });

              {
//                TaskTimer tt("Preparing %d jobs, span %s", jobs.size (), span.toString ().c_str ());

                for (const UpdateQueue::Job& job : jobs)
                  {
                    if (isInterruptionRequested ())
                        break;
                    if (!job.updatejob)
                        continue;

                    if (auto bujob = dynamic_cast<const BlockUpdater::Job*>(job.updatejob.get ()))
                      {
                        auto drawable = block_updater.processJob (*bujob, job.intersecting_blocks);
                        pDrawableChunk d(new ChunkToBlockDegenerateTexture::DrawableChunk(move(drawable)));

//                        chunks_with_blocks.push_back (chunk_with_blocks_t(d, job.intersecting_blocks));
                        for (pBlock b : job.intersecting_blocks)
                            chunks_per_block[b].push(d);
                      }

                    if (auto bujob = dynamic_cast<const TfrMappings::WaveformBlockUpdater::Job*>(job.updatejob.get ()))
                      {
                        TfrMappings::WaveformBlockUpdater().processJob (*bujob, job.intersecting_blocks);
                      }
                  }
              }

            if (!chunks_per_block.empty ())
              {
//                TaskTimer tt("Updating %d blocks", chunks_per_block.size ());
                // draw all chunks who share the same block in one go
                for (map<pBlock, queue<pDrawableChunk>>::value_type& p : chunks_per_block)
                  {
                    if (isInterruptionRequested ())
                        break;

                    shared_ptr<BlockFbo> fbo = block_fbos[p.first];
                    if (!fbo)
                        continue;

//                    TaskTimer tt(boost::format("Drawing %d chunks into block %s")
//                                 % p.second.size() % p.first->getRegion());

                    fbo->begin ();

                    while (!p.second.empty ())
                      {
                        pDrawableChunk c {move(p.second.front ())};
                        p.second.pop ();
                        c->draw ();
                      }

                    fbo->end ();
                  }
              }

//            if (!chunks_with_blocks.empty ())
//              {
//                TaskTimer tt("Updating from %d chunks", chunks_with_blocks.size ());
//                for (auto& c : chunks_with_blocks)
//                  {
//                    for (auto& b : c.second)
//                      {
//                        shared_ptr<BlockFbo> fbo = block_fbos[b];
//                        if (!fbo)
//                            continue;
//                        fbo->begin ();
//                        c.first->draw();
//                        fbo->end ();
//                      }
//                  }
//              }

//            chunks_with_blocks.clear ();

            for (UpdateQueue::Job& j : jobs)
                j.promise.set_value ();

            if (!isInterruptionRequested ())
              {
//                TaskTimer tt("sync");
                block_updater.sync ();
                emit didUpdate ();
              }

            if (false)
            TaskInfo("Updated %d -> %d. %s. %.0f samples/s",
                     jobs.size (), chunks_per_block.size (),
                     span.toString ().c_str (),
                     span.count ()/t.elapsed ());
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
