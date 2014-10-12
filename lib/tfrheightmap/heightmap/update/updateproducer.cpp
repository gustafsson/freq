#include "updateproducer.h"

#include "heightmap/collection.h"
#include "heightmap/blockquery.h"

#include "tfr/chunk.h"

#include "cpumemorystorage.h"
#include "demangle.h"
#include "log.h"
#include "tasktimer.h"

#include <boost/foreach.hpp>


//#define DEBUG_INFO
#define DEBUG_INFO if(0)

using namespace boost;

namespace Heightmap {
namespace Update {

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
    if (pchunk.channel >= (int)C.size())
        return; // ignore, this happens when loading a new file while old things are still being processed

    EXCEPTION_ASSERT_LESS(pchunk.channel, (int)C.size());
    EXCEPTION_ASSERT_LESS_OR_EQUAL(0, pchunk.channel);

    BlockCache::ptr cache = C[pchunk.channel].raw ()->cache();
    Signal::Interval chunk_interval = pchunk.chunk->getCoveredInterval();
    std::vector<pBlock> intersecting_blocks = BlockQuery(cache).getIntersectingBlocks( chunk_interval, false, 0);

    if (intersecting_blocks.empty ())
    {
        Log("Discarding chunk since there are no longer any intersecting_blocks with %s")
                 % chunk_interval;
        return;
    }

    DEBUG_INFO TaskTimer tt(boost::format("updateproducer: channel %d. %s updating %s, %s")
                            % pchunk.channel % pchunk.t->transformDesc ()->toString ()
                            % chunk_interval % vartype(*merge_chunk_.get ()));
    std::vector<std::future<void>> F;

    for (Update::IUpdateJob::ptr job : merge_chunk_->prepareUpdate (pchunk, intersecting_blocks))
    {
        // Use same or different intersecting_blocks
//        intersecting_blocks = BlockQuery(cache).getIntersectingBlocks( job->getCoveredInterval (), false, 0);
//        job->getCoveredInterval ();

        auto f = update_queue_->push( job, intersecting_blocks );
        F.push_back (std::move(f));
    }

    // Wait for these to finish
    // If this worker thread doesn't wait it might produce jobs faster than
    // they can be consumed.
    try
    {
        for (std::future<void>& f : F)
            f.get();
    }
    catch (const std::logic_error&)
    {
        pchunk.abort = true;
        // The queue may be emptied before the task has been processed
        //Log("Discarded job: %s") % chunk_interval;
    }

    // The target view will be refreshed when a job is finished
}


void UpdateProducer::
        set_number_of_channels(unsigned)
{
    // whatever
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


QString UpdateProducerDesc::
        toString() const
{
    return ("View " + transformDesc()->toString ()).c_str();
}

} // namespace Update
} // namespace Heightmap

#include "signal/computingengine.h"
#include "tfr/stft.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

namespace Heightmap {
namespace Update {

class UpdateJobMock : public Update::IUpdateJob {
public:
    UpdateJobMock(bool& called) : called(called) {}

    Signal::Interval getCoveredInterval() const override {
        called = true;
        return Signal::Interval();
    }

    bool& called;
};

class MergeChunkMock : public MergeChunk {
public:
    MergeChunkMock() : chunk_to_block_called(false) {}

    std::vector<Update::IUpdateJob::ptr> prepareUpdate(Tfr::ChunkAndInverse& chunk)
    {
        calledi |= chunk.chunk->getInterval ();

        Update::IUpdateJob::ptr p(new UpdateJobMock(chunk_to_block_called));
        return std::vector<Update::IUpdateJob::ptr> {p};
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
        UpdateQueue::ptr update_queue(new UpdateQueue::ptr::element_type);
        UpdateProducer cbf( update_queue, tfrmap, merge_chunk );

        Tfr::StftDesc stftdesc;
        stftdesc.enable_inverse (false);
        Signal::Interval data = stftdesc.requiredInterval (Signal::Interval(0,4), 0);
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()));

        {
            auto c = tfrmap.read ()->collections()[0];
            Reference entireHeightmap = c->entireHeightmap();
            c->getBlock (entireHeightmap);
        }

        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        EXCEPTION_ASSERT( !merge_chunk_mock->called() );

        std::thread t([&](){cbf(cai);});
        UpdateQueue::Job j = update_queue->pop ();
        j.promise.set_value();
        t.join();

        EXCEPTION_ASSERT( merge_chunk_mock->called() );
        EXCEPTION_ASSERT( j.updatejob );
        EXCEPTION_ASSERT( j );
        EXCEPTION_ASSERT( dynamic_cast<UpdateJobMock*>(j.updatejob.get ()) );
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

        UpdateQueue::ptr update_queue(new UpdateQueue::ptr::element_type);
        UpdateProducerDesc cbfd( update_queue, tfrmap );

        Tfr::pChunkFilter cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( !cf );

        cbfd.setMergeChunkDesc (MergeChunkDesc::ptr( new MergeChunkDescMock ));
        cf = cbfd.createChunkFilter (0);
        EXCEPTION_ASSERT( cf );
        EXCEPTION_ASSERT_EQUALS( vartype(*cf), "Heightmap::Update::UpdateProducer" );

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

} // namespace Update
} // namespace Heightmap
