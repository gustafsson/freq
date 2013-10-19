#include "chunkblockfilter.h"

#include "chunktoblock.h"
#include "collection.h"

#include "tfr/chunk.h"

#include "cpumemorystorage.h"

#include <boost/foreach.hpp>

using namespace boost;

namespace Heightmap {

ChunkBlockFilter::
        ChunkBlockFilter(MergeChunk::Ptr merge_chunk, Heightmap::TfrMapping::Ptr tfrmap)
    :
      tfrmap_(tfrmap),
      merge_chunk_(merge_chunk)
{
}


bool ChunkBlockFilter::
        operator()( Tfr::ChunkAndInverse& pchunk )
{
    TfrMapping::pCollection collection = read1(tfrmap_)->collections()[pchunk.channel];

    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = write1(collection)->getIntersectingBlocks( chunk_interval, false );

    BOOST_FOREACH( pBlock block, intersecting_blocks)
    {
        BlockData::WritePtr blockdata(block->block_data());

        write1(merge_chunk_)->mergeChunk( *block, pchunk, *blockdata );

        blockdata->cpu_copy->OnlyKeepOneStorage<CpuMemoryStorage>();
    }

    return false;
}


ChunkBlockFilterDesc::
        ChunkBlockFilterDesc(Heightmap::TfrMapping::Ptr tfrmap)
    :
      tfrmap_(tfrmap)
{

}


Tfr::pChunkFilter ChunkBlockFilterDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    MergeChunk::Ptr merge_chunk;
    if (merge_chunk_desc_)
        merge_chunk = read1(merge_chunk_desc_)->createMergeChunk(engine);

    if (!merge_chunk)
        return Tfr::pChunkFilter();

    return Tfr::pChunkFilter( new ChunkBlockFilter(merge_chunk, tfrmap_));
}

} // namespace Heightmap

#include "signal/computingengine.h"
#include "tfr/stft.h"

namespace Heightmap {

class MergeChunkMock : public MergeChunk {
public:
    void mergeChunk(
            const Heightmap::Block&,
            const Tfr::ChunkAndInverse& chunk,
            Heightmap::BlockData&)
    {
        called |= chunk.chunk->getInterval ();
    }

    Signal::Intervals called;
};


class MergeChunkDescMock : public MergeChunkDesc {
    MergeChunk::Ptr createMergeChunk(Signal::ComputingEngine* engine) const {
        MergeChunk::Ptr r;
        if (0 == engine) {
            r.reset (new MergeChunkMock());
        }
        return r;
    }
};


void ChunkBlockFilter::
        test()
{
    // It should use a MergeChunk to update all blocks in a tfrmap that matches a given Tfr::Chunk.
    {
        MergeChunkMock* merge_chunk_mock;
        MergeChunk::Ptr merge_chunk( merge_chunk_mock = new MergeChunkMock );
        BlockLayout bl(4, 4, SampleRate(4));
        Heightmap::TfrMapping::Ptr tfrmap(new Heightmap::TfrMapping(bl, ChannelCount(1)));
        write1(tfrmap)->length( 1 );
        ChunkBlockFilter cbf( merge_chunk, tfrmap );

        Tfr::StftDesc stftdesc;
        stftdesc.enable_inverse (false);
        Signal::Interval data = stftdesc.requiredInterval (Signal::Interval(0,4), 0);
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()));

        {
            Heightmap::Collection::WritePtr c(read1(tfrmap)->collections()[0]);
            c->getBlock (c->entireHeightmap ());
        }

        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.inverse = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        cbf(cai);

        EXCEPTION_ASSERT( merge_chunk_mock->called );
    }
}


void ChunkBlockFilterDesc::
        test()
{
    // It should instantiate ChunkBlockFilters for different engines.
    {
        BlockLayout bl(4,4,4);
        Heightmap::TfrMapping::Ptr tfrmap(new Heightmap::TfrMapping(bl, 1));

        ChunkBlockFilterDesc cbfd( tfrmap );

        Tfr::pChunkFilter cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( !cf );

        cbfd.setMergeChunkDesc (MergeChunkDesc::Ptr( new MergeChunkDescMock ));
        cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( cf );
        EXCEPTION_ASSERT_EQUALS( vartype(*cf), "Heightmap::ChunkBlockFilter" );

        Signal::ComputingCpu cpu;
        cf = cbfd.createChunkFilter (&cpu);
        EXCEPTION_ASSERT( !cf );

        Signal::ComputingCuda cuda;
        cf = cbfd.createChunkFilter (&cuda);
        EXCEPTION_ASSERT( !cf );

        Signal::ComputingOpenCL opencl;
        cf = cbfd.createChunkFilter (&opencl);
        EXCEPTION_ASSERT( !cf );
    }
}

} // namespace Heightmap
