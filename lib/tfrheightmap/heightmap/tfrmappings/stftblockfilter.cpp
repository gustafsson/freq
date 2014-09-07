#include "stftblockfilter.h"

#include "heightmap/update/tfrblockupdater.h"
#include "tfr/stft.h"
#include "signal/computingengine.h"

#include "demangle.h"
#include "trace_perf.h"

namespace Heightmap {
namespace TfrMappings {

StftBlockFilter::
        StftBlockFilter(StftBlockFilterParams::ptr params)
    :
      params_(params)
{

}


std::vector<Update::IUpdateJob::ptr> StftBlockFilter::
        prepareUpdate(Tfr::ChunkAndInverse& chunk)
{
    Tfr::StftChunk* stftchunk = dynamic_cast<Tfr::StftChunk*>(chunk.chunk.get ());
    EXCEPTION_ASSERT( stftchunk );

    if (params_) {
        Tfr::pChunkFilter freq_normalization = params_.read ()->freq_normalization;
        if (freq_normalization)
            (*freq_normalization)(chunk);
    }

    float normalization_factor = 1.f/sqrtf(stftchunk->window_size());
    Update::IUpdateJob::ptr chunktoblockp(new Update::TfrBlockUpdater::Job{chunk.chunk, normalization_factor, 0});

    return std::vector<Update::IUpdateJob::ptr>{chunktoblockp};
}


StftBlockFilterDesc::
        StftBlockFilterDesc(StftBlockFilterParams::ptr params)
    :
      params_(params)
{

}


MergeChunk::ptr StftBlockFilterDesc::
        createMergeChunk( Signal::ComputingEngine* engine ) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return MergeChunk::ptr( new StftBlockFilter(params_) );

    return MergeChunk::ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"
#include "detectgdb.h"
#include "heightmap/render/blocktextures.h"

#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

namespace Heightmap {
namespace TfrMappings {

void StftBlockFilter::
        test()
{
    std::string name = "StftBlockFilter";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv); // takes 0.4 s if this is the first instantiation of QApplication
    QGLWidget w;
    w.makeCurrent ();

    // It should update a block with stft transform data.
    {
        TRACE_PERF ("StftBlockFilter should update a block with stft transform data");

        Tfr::StftDesc stftdesc;
        Signal::Interval data = stftdesc.requiredInterval (Signal::Interval(0,4), 0);

        // Create some data to plot into the block
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()/4));
        float *p = buffer->waveform_data()->getCpuMemory ();
        srand(0);
        for (unsigned i=0; i<buffer->getInterval ().count (); ++i) {
            p[i] = -1.f + 2.f*rand()/RAND_MAX;
        }

        // Create a block to plot into
        BlockLayout bl(4,4, buffer->sample_rate ());
        VisualizationParams::ptr vp(new VisualizationParams);
        Heightmap::FreqAxis fa; fa.setLinear (bl.sample_rate ());
        vp->display_scale (fa);

        Reference ref = [&]() {
            Reference ref;
            Position max_sample_size;
            max_sample_size.time = 2.f*std::max(1.f, buffer->length ())/bl.texels_per_row ();
            max_sample_size.scale = 1.f/bl.texels_per_column ();
            ref.log2_samples_size = Reference::Scale(
                        floor_log2( max_sample_size.time ),
                        floor_log2( max_sample_size.scale ));
            ref.block_index = Reference::Index(0,0);
            return ref;
        }();

        Render::BlockTextures::Scoped bt(4,4,1);
        pBlock block( new Block(ref, bl, vp));

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        // Do the merge
        Heightmap::MergeChunk::ptr mc( new StftBlockFilter(StftBlockFilterParams::ptr()) );

        std::queue<Update::UpdateQueue::Job> jobs;

        for (Update::IUpdateJob::ptr job : mc->prepareUpdate (cai))
        {
            Update::UpdateQueue::Job uj;
            uj.intersecting_blocks = std::vector<pBlock>{block};
            uj.updatejob = job;
            jobs.push (std::move(uj));
        }

        Update::TfrBlockUpdater().processJobs (jobs);
        EXCEPTION_ASSERT_EQUALS(jobs.size (), 0u);
    }
}


void StftBlockFilterDesc::
        test()
{
    // It should instantiate StftBlockFilter for different engines.
    {
        Heightmap::MergeChunkDesc::ptr mcd(new StftBlockFilterDesc(StftBlockFilterParams::ptr()));
        MergeChunk::ptr mc = mcd.read ()->createMergeChunk (0);

        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCpu cpu;
        mc = mcd.read ()->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc.get ()), "Heightmap::TfrMappings::StftBlockFilter" );

        Signal::ComputingCuda cuda;
        mc = mcd.read ()->createMergeChunk (&cuda);
        EXCEPTION_ASSERT( !mc );

        Signal::ComputingOpenCL opencl;
        mc = mcd.read ()->createMergeChunk (&opencl);
        EXCEPTION_ASSERT( !mc );
    }
}


} // namespace TfrMappings
} // namespace Heightmap
