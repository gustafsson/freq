#include "chunkmerger.h"

#include "cpumemorystorage.h"
#include "TaskTimer.h"

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
        addChunk( MergeChunk::Ptr mergechunk,
                  Tfr::ChunkAndInverse pchunk,
                  std::vector<pBlock> intersecting_blocks ) volatile
{
    Job j;
    j.merge_chunk = mergechunk;
    j.pchunk = pchunk;
    j.intersecting_blocks = intersecting_blocks;
    WritePtr(this)->jobs.push_back (j);
}


void ChunkMerger::
        processChunks() volatile
{
    WritePtr self(this);

    BOOST_FOREACH(Job& j, self->jobs) {
        processJob(j);
    }

    self->jobs.clear ();
}


void ChunkMerger::
        processJob(Job& j)
{
    write1(j.merge_chunk)->prepareChunk( j.pchunk );

    BOOST_FOREACH( pBlock block, j.intersecting_blocks)
    {
        {
            BlockData::WritePtr blockdata(block->block_data());

            INFO TaskTimer tt(boost::format("chunk %s -> block %s") % j.pchunk.input->getInterval () % block->getRegion ());
            write1(j.merge_chunk)->mergeChunk( *block, j.pchunk, *blockdata );
            blockdata->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();
        }
        block->discard_new_data_available ();
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
