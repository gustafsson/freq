#include "waveformblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "tfr/drawnwaveform.h"
#include "signal/computingengine.h"
#include "tfr/drawnwaveformkernel.h"

#include "demangle.h"

namespace Heightmap {
namespace TfrMappings {

class WaveformChunkToBlock: public IChunkToBlock
{
public:
    WaveformChunkToBlock(Signal::pMonoBuffer b) : b(b) {}

    void mergeChunk( pBlock block ) {
        float blobsize = b->sample_rate() / block->sample_rate();

        int readstop = b->number_of_samples ();

        // todo should substract blobsize/2
        Region r = block->getRegion ();
        float writeposoffs = (b->start () - r.a.time)*block->sample_rate ();
        float y0 = r.a.scale*2-1;
        float yscale = r.scale ()*2;
        ::drawWaveform(
                b->waveform_data(),
                block->block_data ()->cpu_copy,
                blobsize,
                readstop,
                yscale,
                writeposoffs,
                y0);
    }

private:
    Signal::pMonoBuffer b;
};


std::vector<IChunkToBlock::Ptr> WaveformBlockFilter::
        createChunkToBlock(Tfr::ChunkAndInverse& chunk)
{
    IChunkToBlock::Ptr ctb(new WaveformChunkToBlock(chunk.input));
    std::vector<IChunkToBlock::Ptr> R;
    R.push_back (ctb);
    return R;
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

        Heightmap::pBlock block( new Heightmap::Block(ref, bl, vp));
        DataStorageSize s(bl.texels_per_row (), bl.texels_per_column ());
        block->block_data ()->cpu_copy.reset( new DataStorage<float>(s) );

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.input = buffer;

        // Do the merge
        Heightmap::MergeChunk::Ptr mc( new WaveformBlockFilter );
        mc.write ()->filterChunk(cai);
        mc.write ()->createChunkToBlock(cai)[0]->mergeChunk (block);

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
        MergeChunk::Ptr mc = mcd.read ()->createMergeChunk (0);

        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCpu cpu;
        mc = mcd.read ()->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc.get ()), "Heightmap::TfrMappings::WaveformBlockFilter" );

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
