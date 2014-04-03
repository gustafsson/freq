#include "chunkblockfilter.h"

#include "chunktoblock.h"
#include "collection.h"
#include "blockquery.h"

#include "tfr/chunk.h"
#include "blocks/ichunkmerger.h"

#include "cpumemorystorage.h"
#include "demangle.h"

#include <boost/foreach.hpp>

using namespace boost;

namespace Heightmap {

ChunkBlockFilter::
        ChunkBlockFilter( Blocks::IChunkMerger::ptr chunk_merger, Heightmap::TfrMapping::const_ptr tfrmap, MergeChunk::ptr merge_chunk )
    :
      chunk_merger_(chunk_merger),
      tfrmap_(tfrmap),
      merge_chunk_(merge_chunk)
{
}


void ChunkBlockFilter::
        operator()( Tfr::ChunkAndInverse& pchunk )
{
    Heightmap::TfrMapping::Collections C = tfrmap_.read ()->collections();
    EXCEPTION_ASSERT_LESS(pchunk.channel, (int)C.size());
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0, pchunk.channel);

    merge_chunk_->filterChunk( pchunk );

    BlockCache::ptr cache = C[pchunk.channel].read ()->cache();
    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = BlockQuery(cache).getIntersectingBlocks( chunk_interval, false, 0);

    chunk_merger_->addChunk( merge_chunk_, pchunk, intersecting_blocks );
    // The target view will be refreshed when a task is finished, thus calling chunk_merger->processChunks();
}


void ChunkBlockFilter::
        set_number_of_channels(unsigned C)
{
    EXCEPTION_ASSERT_EQUALS((int)C, tfrmap_.read ()->channels());
}


ChunkBlockFilterDesc::
        ChunkBlockFilterDesc( Blocks::IChunkMerger::ptr chunk_merger, Heightmap::TfrMapping::const_ptr tfrmap )
    :
      chunk_merger_(chunk_merger),
      tfrmap_(tfrmap)
{

}


Tfr::pChunkFilter ChunkBlockFilterDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    MergeChunk::ptr merge_chunk;
    if (merge_chunk_desc_)
        merge_chunk = merge_chunk_desc_.read ()->createMergeChunk(engine);

    if (!merge_chunk)
        return Tfr::pChunkFilter();

    return Tfr::pChunkFilter( new ChunkBlockFilter(chunk_merger_, tfrmap_, merge_chunk));
}

} // namespace Heightmap

#include "signal/computingengine.h"
#include "tfr/stft.h"
#include "blocks/chunkmerger.h"
#include <QApplication>
#include <QGLWidget>

namespace Heightmap {

class ChunkToBlockMock : public IChunkToBlock {
public:
    ChunkToBlockMock(bool* called) : called(called) {}

    void init() {}
    void prepareTransfer() {}
    void prepareMerge(AmplitudeAxis amplitude_axis, Tfr::FreqAxis display_scale, BlockLayout bl) {}

    void mergeChunk( pBlock block ) {
        *called = true;
    }

    bool* called;
};

class MergeChunkMock : public MergeChunk {
public:
    MergeChunkMock() : chunk_to_block_called(false) {}

    std::vector<IChunkToBlock::ptr> createChunkToBlock(Tfr::ChunkAndInverse& chunk)
    {
        calledi |= chunk.chunk->getInterval ();

        std::vector<IChunkToBlock::ptr> R;
        IChunkToBlock::ptr p(new ChunkToBlockMock(&chunk_to_block_called));
        R.push_back (p);
        return R;
    }

    bool chunk_to_block_called;
    Signal::Intervals calledi;

    bool called() { return chunk_to_block_called && calledi; }
};


class MergeChunkDescMock : public MergeChunkDesc {
    MergeChunk::ptr createMergeChunk(Signal::ComputingEngine* engine) const {
        MergeChunk::ptr r;
        if (0 == engine) {
            r = MergeChunk::ptr(new MergeChunkMock());
        }
        return r;
    }
};


void ChunkBlockFilter::
        test()
{
    std::string name = "ChunkBlockFilter";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should use a MergeChunk to update all blocks in a tfrmap that matches a given Tfr::Chunk.
    {
        MergeChunkMock* merge_chunk_mock;
        MergeChunk::ptr merge_chunk( merge_chunk_mock = new MergeChunkMock );
        BlockLayout bl(4, 4, SampleRate(4));
        Heightmap::TfrMapping::ptr tfrmap(new Heightmap::TfrMapping(bl, ChannelCount(1)));
        tfrmap.write ()->length( 1 );
        Blocks::IChunkMerger::ptr chunk_merger(new Blocks::ChunkMerger);
        ChunkBlockFilter cbf( chunk_merger, tfrmap, merge_chunk );

        Tfr::StftDesc stftdesc;
        stftdesc.enable_inverse (false);
        Signal::Interval data = stftdesc.requiredInterval (Signal::Interval(0,4), 0);
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()));

        {
            auto c = tfrmap.read ()->collections()[0].read ();
            Reference entireHeightmap = c->entireHeightmap();
            c->getBlock (entireHeightmap);
        }

        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        cbf(cai);

        EXCEPTION_ASSERT( !merge_chunk_mock->called() );

        chunk_merger->processChunks(-1);

        EXCEPTION_ASSERT( merge_chunk_mock->called() );
    }
}


void ChunkBlockFilterDesc::
        test()
{
    std::string name = "ChunkBlockFilterDesc";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should instantiate ChunkBlockFilters for different engines.
    {
        BlockLayout bl(4,4,4);
        Heightmap::TfrMapping::ptr tfrmap(new Heightmap::TfrMapping(bl, 1));

        Blocks::IChunkMerger::ptr chunk_merger(new Blocks::ChunkMerger);
        ChunkBlockFilterDesc cbfd( chunk_merger, tfrmap );

        Tfr::pChunkFilter cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( !cf );

        cbfd.setMergeChunkDesc (MergeChunkDesc::ptr( new MergeChunkDescMock ));
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
