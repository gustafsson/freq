#include "blockupdater.h"
#include "heightmap/update/tfrblockupdater.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"
#include "log.h"
#include "neat_math.h"

#include <thread>
#include <future>

//#define INFO
#define INFO if(0)

using namespace std;

namespace Heightmap {
namespace Update {
namespace OpenGL {


BlockUpdater::
        BlockUpdater()
    :
//      memcpythread(std::thread::hardware_concurrency ())
      memcpythread(2)
{
}


BlockUpdater::
        ~BlockUpdater()
{
    sync ();
}


void BlockUpdater::
        processJobs( const std::vector<UpdateQueue::Job>& jobs )
{
    typedef shared_ptr<ChunkToBlockDegenerateTexture::DrawableChunk> pDrawableChunk;
    ChunkToBlockDegenerateTexture::BlockFbos& block_fbos = chunktoblock_texture.block_fbos ();

    // 'jobs' contains all pBlock and survives longer than pDrawableChunk
    std::map<pBlock, std::queue<pDrawableChunk>> chunks_per_block;

    {
//                TaskTimer tt("Preparing %d jobs, span %s", jobs.size (), span.toString ().c_str ());

      for (const UpdateQueue::Job& job : jobs)
        {
          if (!job.updatejob)
              continue;

          if (auto bujob = dynamic_cast<const TfrBlockUpdater::Job*>(job.updatejob.get ()))
            {
              auto drawable = processJob (*bujob, job.intersecting_blocks);
              pDrawableChunk d(new ChunkToBlockDegenerateTexture::DrawableChunk(move(drawable)));

//                        chunks_with_blocks.push_back (chunk_with_blocks_t(d, job.intersecting_blocks));
              for (pBlock b : job.intersecting_blocks)
                  chunks_per_block[b].push(d);
            }
        }
    }

  if (!chunks_per_block.empty ())
    {
//                TaskTimer tt("Updating %d blocks", chunks_per_block.size ());
      // draw all chunks who share the same block in one go
      for (map<pBlock, queue<pDrawableChunk>>::value_type& p : chunks_per_block)
        {
          shared_ptr<BlockFbo> fbo = block_fbos[p.first];
          if (!fbo)
              continue;

//                    TaskTimer tt(boost::format("Drawing %d chunks into block %s")
//                                 % p.second.size() % p.first->getRegion());

          auto fboScopeBinding = fbo->begin ();

          auto& q = p.second;
          while (!q.empty ())
            {
              pDrawableChunk c {move(q.front ())};
              q.pop ();
              c->draw ();
            }

          // unbind fbo on out-of-scope
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

  sync (); // Wait for block textures to become updated by BlockFbo destructor.
}


ChunkToBlockDegenerateTexture::DrawableChunk BlockUpdater::
        processJob( const TfrBlockUpdater::Job& job,
                    const std::vector<pBlock>& intersecting_blocks )
{
    EXCEPTION_ASSERT (!intersecting_blocks.empty ());
    EXCEPTION_ASSERT (job.chunk);

    pBlock example_block = intersecting_blocks.front ();
    BlockLayout block_layout = example_block->block_layout ();
    Tfr::FreqAxis display_scale = example_block->visualization_params ()->display_scale ();
    AmplitudeAxis amplitude_axis = example_block->visualization_params ()->amplitude_axis ();

    chunktoblock_texture.setParams( amplitude_axis, display_scale, block_layout, job.normalization_factor );

//    TaskTimer tt0(boost::format("processJob %s") % job.getCoveredInterval ());
    // TODO refactor to do one thing at a time
    // 1. prepare to draw from chunks (i.e copy to OpenGL texture and create vertex buffers)
    // 1.1. Same chunk_scale and display_scale for all chunks and all blocks
    // 2. For each block.
    // 2.1. prepare to draw into block
    // 2.2. draw all chunks
    // 2.3. update whatever needs to be updated
    // And rename all these "chunk block merger to merge block chunk merge"

    // Setup FBO
    for (pBlock block : intersecting_blocks)
        chunktoblock_texture.prepareBlock (block);

    // Update PBO and VBO
    auto d = chunktoblock_texture.prepareChunk (job.chunk);
    memcpythread.addTask (d.transferData(job.p));
    return d;
}


void BlockUpdater::
        sync ()
{
    chunktoblock_texture.finishBlocks ();
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap


namespace Heightmap {
namespace Update {
namespace OpenGL {

void BlockUpdater::
        test()
{
    // It should update blocks with chunk data
    {

    }
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
