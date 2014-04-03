#include "cepstrumblockfilter.h"

#include "signal/computingengine.h"
#include "tfr/cepstrum.h"
#include "tfr/stft.h"
#include "heightmap/chunktoblock.h"

#include "demangle.h"

namespace Heightmap {
namespace TfrMappings {

CepstrumBlockFilter::
        CepstrumBlockFilter(CepstrumBlockFilterParams::ptr params)
    :
      params_(params)
{

}


void CepstrumBlockFilter::
        filterChunk(Tfr::ChunkAndInverse&)
{
//    if (params_) {
//        params_.write ();
//    }
}


std::vector<IChunkToBlock::ptr> CepstrumBlockFilter::
        createChunkToBlock(Tfr::ChunkAndInverse& chunk)
{
    Tfr::StftChunk* cepstrumchunk = dynamic_cast<Tfr::StftChunk*>(chunk.chunk.get ());
    EXCEPTION_ASSERT( cepstrumchunk );

    Heightmap::ChunkToBlock* chunktoblock;
    IChunkToBlock::ptr chunktoblockp(chunktoblock = new Heightmap::ChunkToBlock(chunk.chunk));
    //IChunkToBlock::Ptr chunktoblockp(new Heightmap::ChunkToBlockTexture);

    chunktoblock->normalization_factor = 1.f; // already normalized when return from Cepstrum.cpp

    std::vector<IChunkToBlock::ptr> R;
    R.push_back (chunktoblockp);
    return R;
}


CepstrumBlockFilterDesc::
        CepstrumBlockFilterDesc(CepstrumBlockFilterParams::ptr params)
    :
      params_(params)
{

}


MergeChunk::ptr CepstrumBlockFilterDesc::
        createMergeChunk( Signal::ComputingEngine* engine ) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return MergeChunk::ptr( new CepstrumBlockFilter(params_) );

    return MergeChunk::ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"
#include "detectgdb.h"


namespace Heightmap {
namespace TfrMappings {

void CepstrumBlockFilter::
        test()
{
    // It should update a block with cepstrum transform data.
    {
        Timer t;

        Tfr::CepstrumDesc cepstrumdesc;
        Signal::Interval data = cepstrumdesc.requiredInterval (Signal::Interval(0,4), 0);

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

        Heightmap::pBlock block( new Heightmap::Block(ref, bl, vp));
        DataStorageSize s(bl.texels_per_row (), bl.texels_per_column ());
        block->block_data ()->cpu_copy.reset( new DataStorage<float>(s) );

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = cepstrumdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        // Do the merge
        Heightmap::MergeChunk::ptr mc( new CepstrumBlockFilter(CepstrumBlockFilterParams::ptr()) );
        mc->filterChunk(cai);
        mc->createChunkToBlock(cai)[0]->mergeChunk (block);

        float T = t.elapsed ();
        if (DetectGdb::is_running_through_gdb ()) {
            EXCEPTION_ASSERT_LESS(T, 3e-3);
        } else {
            EXCEPTION_ASSERT_LESS(T, 1e-3);
        }
    }
}


void CepstrumBlockFilterDesc::
        test()
{
    // It should instantiate CepstrumBlockFilter for different engines.
    {
        Heightmap::MergeChunkDesc::ptr mcd(new CepstrumBlockFilterDesc(CepstrumBlockFilterParams::ptr()));
        MergeChunk::ptr mc = mcd.read ()->createMergeChunk (0);

        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCpu cpu;
        mc = mcd.read ()->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc.get ()), "Heightmap::TfrMappings::CepstrumBlockFilter" );

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
