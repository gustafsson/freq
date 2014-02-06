#include "chunkmerger.h"

#include "TaskTimer.h"
#include "timer.h"

#include <boost/foreach.hpp>

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {

ChunkMerger::
        ChunkMerger()
{
}


void ChunkMerger::
        clear()
{
    while (!jobs.empty ())
        jobs.pop ();
}


void ChunkMerger::
        addChunk( MergeChunk::Ptr merge_chunk,
                  Tfr::ChunkAndInverse chunk,
                  std::vector<pBlock> intersecting_blocks )
{
    Job j;
    j.merge_chunk = merge_chunk;
    j.chunk = chunk;
    j.intersecting_blocks = intersecting_blocks;
    jobs.push (j);
}


bool ChunkMerger::
        processChunks(float timeout) volatile
{
    Timer t;
    while (timeout < 0 || t.elapsed () < timeout) {
        Job job;

        {
            WritePtr self(this);
            if (self->jobs.empty ())
                return true;
            job = self->jobs.front ();
            self->jobs.pop ();
        }

        processJob (job);
    }
    return false;
}


void ChunkMerger::
        processJob(Job& j)
{
    std::vector<IChunkToBlock::Ptr> chunk_to_blocks = write1( j.merge_chunk )->createChunkToBlock( j.chunk );

    BOOST_FOREACH( IChunkToBlock::Ptr chunk_to_block, chunk_to_blocks)
      {
        BOOST_FOREACH( pBlock block, j.intersecting_blocks)
          {
            INFO TaskTimer tt(boost::format("block %s") % block->getRegion ());
            chunk_to_block->mergeChunk (block);
          }
      }
}

} // namespace Blocks
} // namespace Heightmap



namespace Heightmap {
namespace Blocks {

void ChunkMerger::
        test()
{
    // It should merge chunks into blocks.
    {

    }
}

} // namespace Blocks
} // namespace Heightmap
