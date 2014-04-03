#include "blockupdater.h"

#include "tasktimer.h"
#include "timer.h"

#include <boost/foreach.hpp>

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {

BlockUpdater::
        BlockUpdater()
{

}


void BlockUpdater::
        processJob( MergeChunk::ptr merge_chunk,
                    Tfr::ChunkAndInverse chunk,
                    std::vector<pBlock> intersecting_blocks)
{
    TaskTimer tt0("processJob");
    // TODO refactor to do one thing at a time
    // 1. prepare to draw from chunks (i.e copy to OpenGL texture and create vertex buffers)
    // 1.1. Same chunk_scale and display_scale for all chunks and all blocks
    // 2. For each block.
    // 2.1. prepare to draw into block
    // 2.2. draw all chunks
    // 2.3. update whatever needs to be updated
    // And rename all these "chunk block merger to merge block chunk merge"

    std::vector<IChunkToBlock::ptr> chunk_to_blocks;
    {
//        TaskTimer tt("creating chunk to blocks");
        chunk_to_blocks = merge_chunk->createChunkToBlock( chunk );
    }

    {
//        TaskTimer tt("init");
        for (IChunkToBlock::ptr chunk_to_block : chunk_to_blocks)
            chunk_to_block->init ();
    }
    {
//        TaskTimer tt("prepareTransfer");
        for (IChunkToBlock::ptr chunk_to_block : chunk_to_blocks)
            chunk_to_block->prepareTransfer ();
    }
    // intersecting_blocks is non-empty from addChunk
    pBlock example_block = intersecting_blocks.front ();
    BlockLayout bl = example_block->block_layout ();
    Tfr::FreqAxis display_scale = example_block->visualization_params ()->display_scale ();
    AmplitudeAxis amplitude_axis = example_block->visualization_params ()->amplitude_axis ();

    for (IChunkToBlock::ptr chunk_to_block : chunk_to_blocks)
        chunk_to_block->prepareMerge (amplitude_axis, display_scale, bl);

//    TaskTimer tt(boost::format("processing job %s") % chunk.chunk->getCoveredInterval ());
    for (IChunkToBlock::ptr chunk_to_block : chunk_to_blocks)
      {
        for (pBlock block : intersecting_blocks)
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
