#include "chunktoblock.h"

#include "blockkernel.h"
#include "tfr/chunk.h"

namespace Heightmap {

ChunkToBlock::
        ChunkToBlock()
{

}


void ChunkToBlock::mergeColumnMajorChunk(
        const Block& block,
        const Tfr::Chunk& chunk,
        BlockData& outData )
{
    Region r = block.getRegion();
    VisualizationParams::ConstPtr vp = block.visualization_params ();

    Position chunk_a, chunk_b;
    Signal::Interval inInterval = chunk.getCoveredInterval();
    Signal::Interval blockInterval = block.getInterval();

    // don't validate more texels than we have actual support for
    Signal::Interval spannedBlockSamples(0,0);
    ReferenceInfo ri(block.referenceInfo ());
    Signal::Interval usableInInterval = ri.spannedElementsInterval(inInterval, spannedBlockSamples);

    Signal::Interval transfer = usableInInterval&blockInterval;

    // spannedElementsInterval looks more closely at what in chunk that can be used
    if (!transfer || !spannedBlockSamples)
        return;

    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    // ::resampleStft computes frequency rows properly with its two instances
    // of FreqAxis.
    chunk_a.scale = 0;
    chunk_b.scale = 1;

    ::resampleStft( chunk.transform_data,
                    chunk.nScales(),
                    chunk.nSamples(),
                  outData.cpu_copy,
                  ValidInterval(spannedBlockSamples.first, spannedBlockSamples.last),
                  ResampleArea( chunk_a.time, chunk_a.scale,
                               chunk_b.time, chunk_b.scale ),
                  ResampleArea( r.a.time, r.a.scale,
                               r.b.time, r.b.scale ),
                  chunk.freqAxis,
                  vp->display_scale(),
                  vp->amplitude_axis(),
                  normalization_factor,
                  true);
}


void ChunkToBlock::mergeRowMajorChunk(
        const Block& block,
        const Tfr::Chunk& chunk,
        BlockData& outData )
{
    VisualizationParams::ConstPtr vp = block.visualization_params ();

    // Find out what intervals that match
    Signal::Interval outInterval = block.getInterval();
    Signal::Interval inInterval = chunk.getCoveredInterval();

    // don't validate more texels than we have actual support for
    //Signal::Interval usableInInterval = block->ref.spannedElementsInterval(inInterval);
    Signal::Interval usableInInterval = inInterval;
    Signal::Interval transfer = usableInInterval & outInterval;

    // If block is already up to date, abort merge
    if (!transfer)
        return;

    Region r = block.getRegion();

    float merge_first_scale = r.a.scale;
    float merge_last_scale = r.b.scale;
    float chunk_first_scale = vp->display_scale().getFrequencyScalar( chunk.minHz() );
    float chunk_last_scale = vp->display_scale().getFrequencyScalar( chunk.maxHz() );

    merge_first_scale = std::max( merge_first_scale, chunk_first_scale );
    merge_last_scale = std::min( merge_last_scale, chunk_last_scale );

    if (merge_first_scale >= merge_last_scale)
        return;

    Position chunk_a, chunk_b;
    chunk_a.scale = chunk_first_scale;
    chunk_b.scale = chunk_last_scale;
    chunk_a.time = inInterval.first/chunk.original_sample_rate;
    chunk_b.time = inInterval.last/chunk.original_sample_rate;

    enable_subtexel_aggregation &= full_resolution;

#ifndef CWT_SUBTEXEL_AGGREGATION
    // subtexel aggregation is way to slow
    enable_subtexel_aggregation = false;
#endif

    // Invoke kernel execution to merge chunk into block
    ::blockResampleChunk( chunk.transform_data,
                     outData.cpu_copy,
                     ValidInterval( chunk.first_valid_sample, chunk.first_valid_sample+chunk.n_valid_samples ),
                     //make_uint2( 0, chunk.transform_data->getNumberOfElements().width ),
                     ResampleArea( chunk_a.time, chunk_a.scale,
                                  //chunk_b.time, chunk_b.scale+(chunk_b.scale==1?0.01:0) ), // numerical error workaround, only affects visual
                                 chunk_b.time, chunk_b.scale  ), // numerical error workaround, only affects visual
                     ResampleArea( r.a.time, r.a.scale,
                                  r.b.time, r.b.scale ),
                     complex_info,
                     chunk.freqAxis,
                     vp->display_scale(),
                     vp->amplitude_axis(),
                     normalization_factor,
                     enable_subtexel_aggregation
                     );

}

} // namespace Heightmap

#include "tfr/filter.h"
#include "tfr/stftdesc.h"

namespace Heightmap {

class DummyKernel: public Tfr::ChunkFilter {
    virtual bool operator()( Tfr::ChunkAndInverse& ) {
        return false;
    }
};

class DummyKernelDesc: public Tfr::FilterKernelDesc {
    virtual Tfr::pChunkFilter createChunkFilter(Signal::ComputingEngine* =0) const {
        return Tfr::pChunkFilter(new DummyKernel);
    }
};

void ChunkToBlock::
        test()
{
    ChunkToBlock ctb;
    ctb.complex_info = ComplexInfo_Amplitude_Non_Weighted;
    ctb.enable_subtexel_aggregation = false;
    ctb.full_resolution = false;
    ctb.normalization_factor = 1;
    BlockLayout bl(BlockSize(1<<8,1<<8),100);
    VisualizationParams::Ptr vp(new VisualizationParams);
    TfrMapping tfr_mapping( BlockSize(1<<8,1<<8),100);
    Tfr::FreqAxis ds; ds.setLinear (1);
    vp->display_scale(ds);
    vp->amplitude_axis(AmplitudeAxis_Linear);

    Tfr::StftDesc* tfr;
    Tfr::pTransformDesc tdesc( tfr = new Tfr::StftDesc() );
    Tfr::FilterKernelDesc::Ptr fdesc( new DummyKernelDesc );
    Signal::OperationDesc::Ptr desc(new Tfr::FilterDesc(tdesc, fdesc));
    Signal::Operation::WritePtr operation = write1(read1(desc)->createOperation (0));

    Tfr::TransformKernel* transformkernel = dynamic_cast<Tfr::TransformKernel*>( &*operation );

    Signal::Interval expectedOutput;
    Signal::Interval requiredInterval = read1(desc)->requiredInterval (Signal::Interval (11,31), &expectedOutput);

    Tfr::pTransform t = transformkernel->transform();
    Signal::pMonoBuffer buffer( new Signal::MonoBuffer (requiredInterval, 1));
    Tfr::pChunk chunk = (*t)( buffer );

    Heightmap::Reference ref;

    pBlock block( new Block (ref, bl, vp));
    BlockData blockdata;
    blockdata.cpu_copy.reset( new DataStorage<float>(32,32) );

    ctb.mergeColumnMajorChunk(
            *block,
            *chunk,
            blockdata );

    ctb.mergeRowMajorChunk(
            *block,
            *chunk,
            blockdata );
}

} // namespace Heightmap
