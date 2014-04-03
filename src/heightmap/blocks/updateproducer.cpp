#include "updateproducer.h"

#include "heightmap/chunktoblock.h"
#include "heightmap/collection.h"
#include "heightmap/blockquery.h"

#include "tfr/chunk.h"

#include "cpumemorystorage.h"
#include "demangle.h"

#include <boost/foreach.hpp>

using namespace boost;

namespace Heightmap {
namespace Blocks {

UpdateProducer::
        UpdateProducer( UpdateQueue::ptr update_queue, Heightmap::TfrMapping::const_ptr tfrmap, MergeChunk::ptr merge_chunk )
    :
      update_queue_(update_queue),
      tfrmap_(tfrmap),
      merge_chunk_(merge_chunk)
{
}


void UpdateProducer::
        operator()( Tfr::ChunkAndInverse& pchunk )
{
    Heightmap::TfrMapping::Collections C = tfrmap_.read ()->collections();
    EXCEPTION_ASSERT_LESS(pchunk.channel, (int)C.size());
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0, pchunk.channel);

    merge_chunk_->filterChunk( pchunk );

    BlockCache::ptr cache = C[pchunk.channel].raw ()->cache();
    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = BlockQuery(cache).getIntersectingBlocks( chunk_interval, false, 0);

    TaskTimer tt(boost::format("creating job %s") % chunk_interval);
    update_queue_->addJob( merge_chunk_, pchunk, intersecting_blocks );
    // The target view will be refreshed when a task is finished, thus calling chunk_merger->processChunks();
}


void UpdateProducer::
        set_number_of_channels(unsigned C)
{
    EXCEPTION_ASSERT_EQUALS((int)C, tfrmap_.read ()->channels());
}


UpdateProducerDesc::
        UpdateProducerDesc( UpdateQueue::ptr update_queue, Heightmap::TfrMapping::const_ptr tfrmap )
    :
      update_queue_(update_queue),
      tfrmap_(tfrmap)
{

}


Tfr::pChunkFilter UpdateProducerDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    MergeChunk::ptr merge_chunk;
    if (merge_chunk_desc_)
        merge_chunk = merge_chunk_desc_.read ()->createMergeChunk(engine);

    if (!merge_chunk)
        return Tfr::pChunkFilter();

    return Tfr::pChunkFilter( new UpdateProducer(update_queue_, tfrmap_, merge_chunk));
}

} // namespace Blocks
} // namespace Heightmap

#include "signal/computingengine.h"
#include "tfr/stft.h"
#include "blockupdater.h"

#include <QApplication>
#include <QGLWidget>

namespace Heightmap {
namespace Blocks {

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


void UpdateProducer::
        test()
{
    std::string name = "UpdateProducer";
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
        UpdateQueue::ptr update_queue(new UpdateQueue);
        UpdateProducer cbf( update_queue, tfrmap, merge_chunk );

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

        UpdateQueue::Job j = update_queue->getJob ();
        Blocks::BlockUpdater().processJob(j.merge_chunk, j.chunk, j.intersecting_blocks);

        EXCEPTION_ASSERT( merge_chunk_mock->called() );
    }
}


void UpdateProducerDesc::
        test()
{
    std::string name = "UpdateProducerDesc";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should instantiate UpdateProducer for different engines.
    {
        BlockLayout bl(4,4,4);
        Heightmap::TfrMapping::ptr tfrmap(new Heightmap::TfrMapping(bl, 1));

        UpdateQueue::ptr update_queue(new UpdateQueue);
        UpdateProducerDesc cbfd( update_queue, tfrmap );

        Tfr::pChunkFilter cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( !cf );

        cbfd.setMergeChunkDesc (MergeChunkDesc::ptr( new MergeChunkDescMock ));
        cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( cf );
        EXCEPTION_ASSERT_EQUALS( vartype(*cf), "Heightmap::Blocks::UpdateProducer" );

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

} // namespace Blocks
} // namespace Heightmap
