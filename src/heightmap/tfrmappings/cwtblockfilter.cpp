#include "cwtblockfilter.h"
#include "heightmap/chunktoblock.h"
#include "heightmap/chunktoblockdegeneratetexture.h"
#include "heightmap/chunkblockfilter.h"
#include "tfr/cwtchunk.h"
#include "tfr/cwt.h"
#include "signal/computingengine.h"

#include "demangle.h"

#include <boost/foreach.hpp>

namespace Heightmap {
namespace TfrMappings {

CwtBlockFilter::
        CwtBlockFilter(ComplexInfo complex_info)
    :
      complex_info_(complex_info)
{}


std::vector<IChunkToBlock::Ptr> CwtBlockFilter::
        createChunkToBlock(Tfr::ChunkAndInverse& pchunk)
{
    Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(pchunk.t.get ());
    EXCEPTION_ASSERT( cwt );
//    bool full_resolution = cwt->wavelet_time_support() >= cwt->wavelet_default_time_support();
    float normalization_factor = cwt->nScales()/cwt->sigma();

    Tfr::CwtChunk& chunks = *dynamic_cast<Tfr::CwtChunk*>( pchunk.chunk.get () );

    std::vector<IChunkToBlock::Ptr> R;

    for ( const Tfr::pChunk& chunkpart : chunks.chunks )
      {
//        Heightmap::ChunkToBlock* chunktoblock;
//        IChunkToBlock::Ptr chunktoblockp(chunktoblock = new Heightmap::ChunkToBlock(chunkpart));
//        chunktoblock->full_resolution = full_resolution;
//        chunktoblock->enable_subtexel_aggregation = false; //renderer->redundancy() <= 1;
//        chunktoblock->complex_info = complex_info_;

        // Compute the norm of the chunk prior to resampling and interpolating
        Tfr::ChunkElement *p = chunkpart->transform_data->getCpuMemory ();
        int n = chunkpart->transform_data->numberOfElements ();
        for (int i = 0; i<n; ++i)
            p[i] = Tfr::ChunkElement( norm(p[i]), 0.f );

        IChunkToBlock::Ptr chunktoblockp(new Heightmap::ChunkToBlockDegenerateTexture(chunkpart));
        chunktoblockp->normalization_factor = normalization_factor;
        EXCEPTION_ASSERT_EQUALS( complex_info_, ComplexInfo_Amplitude_Non_Weighted );

        R.push_back (chunktoblockp);
      }

    return R;
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
    if (dynamic_cast<Signal::ComputingCpu*>(engine))
        return MergeChunk::Ptr(new CwtBlockFilter(complex_info_));

    return MergeChunk::Ptr();
}

} // namespace TfrMappings
} // namespace Heightmap


#include "timer.h"
#include "neat_math.h"
#include "signal/computingengine.h"
#include "test/randombuffer.h"

#include <QApplication>
#include <QGLWidget>

namespace Heightmap {
namespace TfrMappings {

void CwtBlockFilter::
        test()
{
    std::string name = "CwtBlockFilter";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    // It should update a block with cwt transform data.
    {
        Timer t;

        Tfr::Cwt cwtdesc;
        float fs = 1024;
        cwtdesc.set_wanted_min_hz(20, fs);
        Signal::Interval i(0,4);
        Signal::Interval expected;
        Signal::Interval data = cwtdesc.requiredInterval (i, &expected);
        EXCEPTION_ASSERT(expected & Signal::Interval(i.first, i.first+1));

        // Create some data to plot into the block
        Signal::pMonoBuffer buffer = Test::RandomBuffer::randomBuffer (data, fs, 1)->getChannel (0);

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

        Heightmap::pBlock block(new Heightmap::Block(ref, bl, vp));
        DataStorageSize s(bl.texels_per_row (), bl.texels_per_column ());
        block->block_data ()->cpu_copy.reset( new DataStorage<float>(s) );

        // Create some data to plot into the block
        Tfr::ChunkAndInverse cai;
        cai.channel = 0;
        cai.input = buffer;
        cai.t = cwtdesc.createTransform ();
        cai.chunk = (*cai.t)( buffer );
        EXCEPTION_ASSERT_EQUALS(cai.chunk->getCoveredInterval (), expected);

        // Do the merge
        ComplexInfo complex_info = ComplexInfo_Amplitude_Non_Weighted;
        Heightmap::MergeChunk::Ptr mc( new CwtBlockFilter(complex_info) );

        write1(mc)->filterChunk(cai);
        std::vector<IChunkToBlock::Ptr> prep = write1(mc)->createChunkToBlock(cai);
        for (size_t i=0; i<prep.size (); ++i)
            prep[i]->mergeChunk (block);

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

        EXCEPTION_ASSERT( !mc );

        Signal::ComputingCpu cpu;
        mc = read1(mcd)->createMergeChunk (&cpu);
        EXCEPTION_ASSERT( mc );
        EXCEPTION_ASSERT_EQUALS( vartype(*mc), "Heightmap::TfrMappings::CwtBlockFilter" );

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
