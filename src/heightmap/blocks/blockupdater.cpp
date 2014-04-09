#include "blockupdater.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"

#include <thread>
#include <future>

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {


BlockUpdater::Job::Job(Tfr::pChunk chunk, float normalization_factor)
    :
      chunk(chunk),
      p(0),
      normalization_factor(normalization_factor)
{
    Tfr::ChunkElement *cp = chunk->transform_data->getCpuMemory ();
    int n = chunk->transform_data->numberOfElements ();
    // Compute the norm of the complex elements in the chunk prior to resampling and interpolating
    float *p = (float*)cp; // Overwrite 'cp'
    // This takes a while, simply because p is large so that a lot of memory has to be copied.
    for (int i = 0; i<n; ++i)
        p[i] = norm(cp[i]); // Compute norm here and square root in shader.

    this->p = p;
    // Keep chunk->transform_data for memory management
}


Signal::Interval BlockUpdater::Job::
        getCoveredInterval() const
{
    return chunk->getCoveredInterval ();
}


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


ChunkToBlockDegenerateTexture::DrawableChunk BlockUpdater::
        processJob( const BlockUpdater::Job& job,
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

} // namespace Blocks
} // namespace Heightmap


namespace Heightmap {
namespace Blocks {

void BlockUpdater::
        test()
{
    // It should update blocks with chunk data
    {

    }
}

} // namespace Blocks
} // namespace Heightmap
