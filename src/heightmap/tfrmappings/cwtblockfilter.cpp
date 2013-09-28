#include "cwtblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/chunkblockfilter.h"
#include "tfr/cwtchunk.h"
#include "tfr/cwt.h"

#include <boost/foreach.hpp>

namespace Heightmap {
namespace TfrMappings {

CwtBlockFilter::
        CwtBlockFilter(ComplexInfo complex_info)
    :
      complex_info_(complex_info)
{}


void CwtBlockFilter::
        mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& pchunk, Heightmap::BlockData& outData )
{
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(pchunk.t.get ());
    EXCEPTION_ASSERT( cwt );
    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();
    float normalization_factor = cwt->nScales( pchunk.chunk->original_sample_rate )/cwt->sigma();

    Tfr::CwtChunk& chunks = *dynamic_cast<Tfr::CwtChunk*>( pchunk.chunk.get () );

    Heightmap::ChunkToBlock chunktoblock;
    chunktoblock.full_resolution = full_resolution;
    chunktoblock.complex_info = complex_info_;
    chunktoblock.normalization_factor = normalization_factor;
    chunktoblock.enable_subtexel_aggregation = false; //renderer->redundancy() <= 1;

    BOOST_FOREACH( const Tfr::pChunk& chunkpart, chunks.chunks )
    {
        chunktoblock.mergeRowMajorChunk (block, *chunkpart, outData);
    }
}


CwtBlockFilterDesc::
        CwtBlockFilterDesc(ComplexInfo complex_info)
    :
      complex_info_(complex_info)
{
}


MergeChunk::Ptr CwtBlockFilterDesc::
        createMergeChunk(Signal::ComputingEngine* engine) const
{
    if (0 == engine)
        return MergeChunk::Ptr(new CwtBlockFilter(complex_info_));

    return MergeChunk::Ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"

namespace Heightmap {
namespace TfrMappings {

void CwtBlockFilter::
        test()
{
    // It should update a block with cwt transform data.
    {
        Timer t;

        Tfr::Cwt cwtdesc;
        cwtdesc.set_fs (1);
        Signal::Interval data = cwtdesc.requiredInterval (Signal::Interval(0,4), 0);

        // Create some data to plot into the block
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()/4));
        float *p = buffer->waveform_data()->getCpuMemory ();
        srand(0);
        for (unsigned i=0; i<buffer->getInterval ().count (); ++i) {
            p[i] = -1.f + 2.f*rand()/RAND_MAX;
        }

        // Create a block to plot into
        TfrMapping tfr_mapping(BlockSize(4,4), buffer->sample_rate ());
        Reference ref; {
            Position max_sample_size;
            max_sample_size.time = 2.f*std::max(1.f, buffer->length ())/tfr_mapping.block_size.texels_per_row ();
            max_sample_size.scale = 1.f/tfr_mapping.block_size.texels_per_column ();
            ref.log2_samples_size = Reference::Scale(
                        floor_log2( max_sample_size.time ),
                        floor_log2( max_sample_size.scale ));
            ref.block_index = Reference::Index(0,0);
        }

        Heightmap::Block block(ref, tfr_mapping);
        DataStorageSize s(tfr_mapping.block_size.texels_per_row (), tfr_mapping.block_size.texels_per_column ());
        write1(block.block_data ())->cpu_copy.reset( new DataStorage<float>(s) );

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.inverse = buffer;
        cai.t = cwtdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );
        ComplexInfo complex_info = ComplexInfo_Amplitude_Non_Weighted;

        // Do the merge
        Heightmap::MergeChunk::Ptr mc( new CwtBlockFilter(complex_info) );
        write1(mc)->mergeChunk( block, cai, *write1(block.block_data ()) );

        float T = t.elapsed ();
        EXCEPTION_ASSERT_LESS(T, 1.0); // this is ridiculously slow
    }
}

void CwtBlockFilterDesc::
        test()
{
    // It should instantiate CwtBlockFilter for different engines.
    {
        ComplexInfo complex_info = ComplexInfo_Amplitude_Non_Weighted;
        Heightmap::MergeChunkDesc::Ptr mcd(new CwtBlockFilterDesc(complex_info));
        MergeChunk::Ptr mc = read1(mcd)->createMergeChunk (0);

        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc), "Heightmap::TfrMappings::CwtBlockFilter" );

        Signal::ComputingCpu cpu;
        mc = read1(mcd)->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCuda cuda;
        mc = read1(mcd)->createMergeChunk (&cuda);
        EXCEPTION_ASSERT( !mc );

        Signal::ComputingOpenCL opencl;
        mc = read1(mcd)->createMergeChunk (&opencl);
        EXCEPTION_ASSERT( !mc );
    }
}

} // namespace TfrMappings
} // namespace Heightmap
