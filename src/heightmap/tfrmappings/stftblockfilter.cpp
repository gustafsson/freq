#include "stftblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/chunkblockfilter.h"
#include "tfr/stft.h"

namespace Heightmap {
namespace TfrMappings {

void StftBlockFilter::
        mergeChunk( const Heightmap::Block& block, const Tfr::ChunkAndInverse& pchunk, Heightmap::BlockData& outData )
{
    Tfr::StftChunk* stftchunk = dynamic_cast<Tfr::StftChunk*>(pchunk.chunk.get ());
    EXCEPTION_ASSERT( stftchunk );
    float normalization_factor = 1.f/sqrtf(stftchunk->window_size());

    Heightmap::ChunkToBlock chunktoblock;
    chunktoblock.normalization_factor = normalization_factor;
    chunktoblock.mergeColumnMajorChunk (block, *pchunk.chunk, outData);
}


MergeChunk::Ptr StftBlockFilterDesc::
        createMergeChunk( Signal::ComputingEngine* engine ) const
{
    if (0 == engine)
        return MergeChunk::Ptr( new StftBlockFilter );

    return MergeChunk::Ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"
#include "detectgdb.h"


namespace Heightmap {
namespace TfrMappings {

void StftBlockFilter::
        test()
{
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
        cai.t = stftdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );

        // Do the merge
        Heightmap::MergeChunk::Ptr mc( new StftBlockFilter );
        write1(mc)->mergeChunk( block, cai, *write1(block.block_data ()) );

        float T = t.elapsed ();
        if (DetectGdb::is_running_through_gdb ()) {
            EXCEPTION_ASSERT_LESS(T, 2e-3);
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
        Heightmap::MergeChunkDesc::Ptr mcd(new StftBlockFilterDesc);
        MergeChunk::Ptr mc = read1(mcd)->createMergeChunk (0);

        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc), "Heightmap::TfrMappings::StftBlockFilter" );

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
