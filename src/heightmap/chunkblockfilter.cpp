#include "chunkblockfilter.h"

#include "chunktoblock.h"
#include "collection.h"

#include "tfr/chunk.h"

#include "cpumemorystorage.h"

#include <boost/foreach.hpp>

using namespace boost;

namespace Heightmap {


///////////////////////////////////////////////////////////////////////////////


ChunkBlockFilterKernel::
        ChunkBlockFilterKernel(Heightmap::TfrMap::Ptr tfrmap)
    :
      tfrmap_(tfrmap)
{
}


bool ChunkBlockFilterKernel::
        applyFilter( Tfr::ChunkAndInverse& pchunk )
{
    Heightmap::ChunkToBlock chunk_to_block;
    chunk_to_block.complex_info = Heightmap::ComplexInfo_Amplitude_Weighted;
    chunk_to_block.normalization_factor = 1;
    chunk_to_block.full_resolution = true;
    chunk_to_block.enable_subtexel_aggregation = false;
    chunk_to_block.tfr_mapping = read1(tfrmap_)->tfr_mapping();

    TfrMap::pCollection collection = read1(tfrmap_)->collections()[pchunk.channel];

    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = write1(collection)->getIntersectingBlocks( chunk_interval, false );

    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
        BlockData::WritePtr blockdata(block->block_data());

        if (pchunk.chunk->order == Tfr::Chunk::Order_row_major)
            chunk_to_block.mergeRowMajorChunk ( *block, pchunk.chunk, *blockdata );
        else
            chunk_to_block.mergeColumnMajorChunk ( *block, pchunk.chunk, *blockdata );

        blockdata->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();

        block->new_data_available = true;
    }

    return false;
}


bool ChunkBlockFilterKernel::
        operator()( Tfr::Chunk& )
{
    EXCEPTION_ASSERTX( false, "Invalid call");

    return false;
}


///////////////////////////////////////////////////////////////////////////////


ChunkBlockFilterKernelDesc::
        ChunkBlockFilterKernelDesc(Heightmap::TfrMap::Ptr tfrmap)
    :
      tfrmap_(tfrmap)
{
}


ChunkBlockFilterKernelDesc::
        ~ChunkBlockFilterKernelDesc()
{
}


Tfr::pChunkFilter ChunkBlockFilterKernelDesc::
        createChunkFilter(Signal::ComputingEngine* /*engine*/) const
{
    return Tfr::pChunkFilter(new ChunkBlockFilterKernel(tfrmap_));
}


///////////////////////////////////////////////////////////////////////////////


Signal::OperationDesc::Ptr CreateChunkBlockFilter::
        createOperationDesc (Heightmap::TfrMap::Ptr tfrmap)
{
    Tfr::TransformDesc::Ptr ttd = read1(tfrmap)->transform_desc();
    Tfr::FilterKernelDesc::Ptr fkdp (new ChunkBlockFilterKernelDesc(tfrmap));
    Signal::OperationDesc::Ptr odp (new Tfr::FilterDesc(ttd, fkdp));
    return odp;
}


void CreateChunkBlockFilter::
        test()
{
    Heightmap::TfrMap::Ptr tfrmap = Heightmap::TfrMap::testInstance ();

    Signal::OperationDesc::Ptr opdesc = createOperationDesc (tfrmap);

    Signal::Operation::Ptr op = opdesc->createOperation ();
}


} // namespace Heightmap
