#include "chunkmerger.h"

#include "cpumemorystorage.h"
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
        addChunk( MergeChunk::Ptr merge_chunk,
                  Tfr::ChunkAndInverse pchunk,
                  std::vector<pBlock> intersecting_blocks ) volatile
{
    write1(merge_chunk)->prepareChunk( pchunk );

    Job j;
    j.merge_chunk = merge_chunk;
    j.pchunk = pchunk;
    j.intersecting_blocks = intersecting_blocks;
    WritePtr(this)->jobs.push (j);
}


void ChunkMerger::
        processChunks(float timeout) volatile
{
    Timer t;
    while (timeout < 0 || t.elapsed () < timeout) {
        Job job;
        {
            WritePtr self(this);
            if (self->jobs.empty ())
                return;
            job = self->jobs.top ();
            self->jobs.pop ();
        }

        processJob (job);
    }
}


void ChunkMerger::
        processJob(Job& j)
{
    BOOST_FOREACH( pBlock block, j.intersecting_blocks)
    {
        BlockData::WritePtr blockdata(block->block_data());

        INFO TaskTimer tt(boost::format("chunk %s -> block %s") % j.pchunk.input->getInterval () % block->getRegion ());
        write1(j.merge_chunk)->mergeChunk( *block, j.pchunk, *blockdata );
        blockdata->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();
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
