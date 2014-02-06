#include "chunkblockfilter.h"

#include "chunktoblock.h"
#include "collection.h"

#include "tfr/chunk.h"
#include "blocks/chunkmerger.h"

#include "cpumemorystorage.h"
#include "demangle.h"

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


void ChunkBlockFilter::
        operator()( Tfr::ChunkAndInverse& pchunk )
{
    Heightmap::TfrMapping::Collections C = read1(tfrmap_)->collections();
    EXCEPTION_ASSERT_LESS(pchunk.channel, (int)C.size());
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0, pchunk.channel);
    TfrMapping::pCollection collection = C[pchunk.channel];

    Blocks::ChunkMerger::Ptr chunk_merger = read1(collection)->chunk_merger();
    write1(merge_chunk_)->filterChunk( pchunk );

    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = write1(collection)->getIntersectingBlocks( chunk_interval, false );

    write1(chunk_merger)->addChunk( merge_chunk_, pchunk, intersecting_blocks );
    // The target view will be refreshed when a task is finished, thus calling chunk_merger->processChunks();
}


void ChunkBlockFilter::
        set_number_of_channels(unsigned C)
{
    write1(tfrmap_)->channels(C);
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
#include <QApplication>
#include <QGLWidget>

namespace Heightmap {

class ChunkToBlockMock : public IChunkToBlock {
public:
    ChunkToBlockMock(bool* called) : called(called) {}

    void mergeChunk( pBlock block ) {
        *called = true;
    }

    bool* called;
};

class MergeChunkMock : public MergeChunk {
public:
    MergeChunkMock() : chunk_to_block_called(false) {}

    std::vector<IChunkToBlock::Ptr> createChunkToBlock(Tfr::ChunkAndInverse& chunk)
    {
        calledi |= chunk.chunk->getInterval ();

        std::vector<IChunkToBlock::Ptr> R;
        IChunkToBlock::Ptr p(new ChunkToBlockMock(&chunk_to_block_called));
        R.push_back (p);
        return R;
    }

    bool chunk_to_block_called;
    Signal::Intervals calledi;

    bool called() { return chunk_to_block_called && calledi; }
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
    std::string name = "ChunkBlockFilter";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

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
        cai.input = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        cbf(cai);

        EXCEPTION_ASSERT( !merge_chunk_mock->called() );

        {
            Heightmap::Collection::ReadPtr c(read1(tfrmap)->collections()[0]);
            c->chunk_merger()->processChunks(-1);
        }

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
