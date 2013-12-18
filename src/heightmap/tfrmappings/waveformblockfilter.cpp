#include "waveformblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "tfr/drawnwaveform.h"
#include "signal/computingengine.h"
#include "tfr/drawnwaveformkernel.h"

namespace Heightmap {
namespace TfrMappings {

WaveformBlockFilter::
        WaveformBlockFilter()
{
}


void WaveformBlockFilter::
        mergeChunk(
            const Heightmap::Block& block,
            const Tfr::ChunkAndInverse& chunk,
            Heightmap::BlockData& outData )
{
    // tfrmap should be
//    Tfr::FreqAxis fa = read1(tfr_map_)->display_scale();
//    Chunk& chunk = *pchunk.chunk;
//    if (fa.min_hz != chunk.freqAxis.min_hz || fa.axis_scale != Tfr::AxisScale_Linear)
//    {
//        EXCEPTION_ASSERT( fa.max_frequency_scalar == 1.f );
//        fa.axis_scale = Tfr::AxisScale_Linear;
//        fa.min_hz = chunk.freqAxis.min_hz;
//        fa.f_step = -2*fa.min_hz;
//        write1(tfr_map_)->display_scale( fa );
//    }

    // updateMaxValue(b);

    Signal::pMonoBuffer b = chunk.inverse;
    float blobsize = b->sample_rate() / block.sample_rate();

    int readstop = b->number_of_samples ();

    float writeposoffs = (b->start () - block.getRegion ().a.time)*block.sample_rate ();
    ::drawWaveform(
            b->waveform_data(),
            outData.cpu_copy,
            blobsize,
            readstop,
            1.f,
            writeposoffs);
}


WaveformBlockFilterDesc::
        WaveformBlockFilterDesc()
{
}


MergeChunk::Ptr WaveformBlockFilterDesc::
        createMergeChunk(Signal::ComputingEngine* engine) const
{
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return MergeChunk::Ptr(new WaveformBlockFilter);

    return MergeChunk::Ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"

namespace Heightmap {
namespace TfrMappings {

void WaveformBlockFilter::
        test()
{
    // It should update a block with cwt transform data.
    {
        Timer t;

//        Tfr::DrawnWaveform waveformdesc;
//        Signal::Interval data = waveformdesc.requiredInterval (Signal::Interval(0,4), 0);

        Signal::Interval data(0,4);
        // Create some data to plot into the block
        Signal::pMonoBuffer buffer(new Signal::MonoBuffer(data, data.count ()/4));
        float *p = buffer->waveform_data()->getCpuMemory ();
        srand(0);
        for (unsigned i=0; i<buffer->getInterval ().count (); ++i) {
            p[i] = -1.f + 2.f*rand()/RAND_MAX;
        }

        // Create a block to plot into
        BlockLayout bl(4,4, buffer->sample_rate ());
        VisualizationParams::Ptr vp(new VisualizationParams);
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

        Heightmap::Block block(ref, bl, vp);
        DataStorageSize s(bl.texels_per_row (), bl.texels_per_column ());
        block.block_data ()->cpu_copy.reset( new DataStorage<float>(s) );

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.inverse = buffer;

        // Do the merge
        Heightmap::MergeChunk::Ptr mc( new WaveformBlockFilter );
        write1(mc)->mergeChunk( block, cai, *block.block_data () );

        float T = t.elapsed ();
        EXCEPTION_ASSERT_LESS(T, 1.0); // this is ridiculously slow
    }
}


void WaveformBlockFilterDesc::
        test()
{
    // It should instantiate CwtBlockFilter for different engines.
    {
        Heightmap::MergeChunkDesc::Ptr mcd(new WaveformBlockFilterDesc);
        MergeChunk::Ptr mc = read1(mcd)->createMergeChunk (0);

        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCpu cpu;
        mc = read1(mcd)->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc), "Heightmap::TfrMappings::WaveformBlockFilter" );

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
