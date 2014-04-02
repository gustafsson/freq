#include "stftblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/chunktoblocktexture.h"
#include "heightmap/chunktoblockdegeneratetexture.h"
#include "heightmap/chunkblockfilter.h"
#include "tfr/stft.h"
#include "signal/computingengine.h"
#include "heightmap/glblock.h"

#include "demangle.h"

namespace Heightmap {
namespace TfrMappings {

StftBlockFilter::
        StftBlockFilter(StftBlockFilterParams::ptr params)
    :
      params_(params)
{

}


void StftBlockFilter::
        filterChunk(Tfr::ChunkAndInverse& chunk)
{
    if (params_) {
        Tfr::pChunkFilter freq_normalization = params_.read ()->freq_normalization;
        if (freq_normalization)
            (*freq_normalization)(chunk);
    }
}


std::vector<IChunkToBlock::ptr> StftBlockFilter::
        createChunkToBlock(Tfr::ChunkAndInverse& chunk)
{
    Tfr::StftChunk* stftchunk = dynamic_cast<Tfr::StftChunk*>(chunk.chunk.get ());
    EXCEPTION_ASSERT( stftchunk );

    IChunkToBlock::ptr chunktoblock;

    try {
        chunktoblock.reset(new Heightmap::ChunkToBlockDegenerateTexture(chunk.chunk));
    } catch (const ExceptionAssert& x) {
        try {
            TaskTimer tt("ChunkToBlockDegenerateTexture failed. Trying CPU fallback");
            chunktoblock.reset(new Heightmap::ChunkToBlock(chunk.chunk));
        } catch(...) {
            throw x;
        }
    }

    chunktoblock->normalization_factor = 1.f/sqrtf(stftchunk->window_size());
    std::vector<IChunkToBlock::ptr> R;
    R.push_back (chunktoblock);
    return R;
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
#include <QApplication>
#include <QGLWidget>

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
        Timer t;

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

        Heightmap::pBlock block(new Heightmap::Block(ref, bl, vp));
        DataStorageSize s(bl.texels_per_row (), bl.texels_per_column ());
        block->block_data ()->cpu_copy.reset( new DataStorage<float>(s) );
        Region r = RegionFactory( bl )( ref );
        block->glblock.reset( new GlBlock( bl, r.time(), r.scale() ));

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        // Do the merge
        Heightmap::MergeChunk::ptr mc( new StftBlockFilter(StftBlockFilterParams::ptr()) );
        mc->filterChunk(cai);
        mc->createChunkToBlock(cai)[0]->mergeChunk (block);

        float T = t.elapsed ();
//        if (DetectGdb::is_running_through_gdb ()) {
//            EXCEPTION_ASSERT_LESS(T, 3e-3);
//        } else {
//            EXCEPTION_ASSERT_LESS(T, 1e-3);
//        }
        if (DetectGdb::is_running_through_gdb ()) {
            EXCEPTION_ASSERT_LESS(T, 50e-3);
        } else {
            EXCEPTION_ASSERT_LESS(T, 1e-3);
        }
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
