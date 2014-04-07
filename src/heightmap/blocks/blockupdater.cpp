#include "blockupdater.h"
#include "tfr/chunk.h"

#include "tasktimer.h"
#include "timer.h"

#include <boost/foreach.hpp>

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {

Signal::Interval BlockUpdater::Job::
        getCoveredInterval() const
{
    return chunk->getCoveredInterval ();
}


BlockUpdater::
        BlockUpdater()
{

}


void BlockUpdater::
        processJob( const BlockUpdater::Job& job,
                    std::vector<pBlock> intersecting_blocks )
{
    if (intersecting_blocks.empty ())
        return; // Nothing to do

    EXCEPTION_ASSERT(job.chunk);

    pBlock example_block = intersecting_blocks.front ();
    BlockLayout block_layout = example_block->block_layout ();
    Tfr::FreqAxis display_scale = example_block->visualization_params ()->display_scale ();
    AmplitudeAxis amplitude_axis = example_block->visualization_params ()->amplitude_axis ();

    chunktoblock_texture.setParams( amplitude_axis, display_scale, block_layout, job.normalization_factor );

    TaskTimer tt0(boost::format("processJob %s") % job.getCoveredInterval ());
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
    auto drawable = chunktoblock_texture.prepareChunk (job.chunk);

//    typedef std::pair<ChunkToBlockDegenerateTexture::DrawableChunk, std::vector<pBlock>> DrawableBlocks;
//    std::vector<DrawableBlocks> chunks;
//    chunks.push_back (DrawableBlocks(std::move(drawable), intersecting_blocks));

    {
        TaskTimer tt("prepareShader");
        drawable.prepareShader ();
    }

    TaskTimer tt2("draw");
    for (pBlock block : intersecting_blocks)
      {
        if (!block->frame_number_last_used)
            continue;

        INFO TaskTimer tt(boost::format("block %s") % block->getRegion ());
        drawable.draw (block);
      }
}


void BlockUpdater::
        sync ()
{
    TaskTimer tt("sync");
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
